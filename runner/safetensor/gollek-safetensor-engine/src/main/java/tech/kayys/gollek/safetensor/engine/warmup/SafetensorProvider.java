/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorProvider.java  (updated — DIRECT backend added)
 * ──────────────────────────────────────────────────────────
 * This file replaces the blueprint version from gollek-safetensor.txt.
 *
 * Changes from the blueprint:
 *   1. Backend enum: added DIRECT
 *   2. @Inject DirectSafetensorBackend directBackend
 *   3. initialize():  initialises directBackend when backend == DIRECT
 *   4. infer():       routes to directBackend.infer() when backend == DIRECT
 *   5. inferStream(): routes to directBackend.inferStream()
 *   6. health():      delegates to directBackend.health()
 *   7. capabilities(): adds "direct-inference" feature flag when DIRECT is active
 *   8. resolveBackend(): adds "direct" case
 *   9. supports():    also accepts model directories (for Hub-downloaded models)
 *   10. metadata():   updated description
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.ConfigProvider;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.spi.SafetensorFeature;

import tech.kayys.gollek.spi.exception.ProviderException;

import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.core.model.ModelFormat;
import tech.kayys.gollek.spi.observability.AdapterMetricTagResolver;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.pipeline.ModelPipeline;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRegistry;
import tech.kayys.gollek.spi.pipeline.ModelPipelineRequest;
import tech.kayys.gollek.spi.provider.*;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelModalityTraits;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;

/**
 * SafeTensors provider with three execution backends:
 *
 * <ul>
 * <li><b>auto</b> — prefer GGUF conversion if available, else DIRECT, else
 * LibTorch.
 * <li><b>gguf</b> — always convert to GGUF and run via llama.cpp.
 * <li><b>direct</b> — run raw safetensors weights directly via
 * DirectInferenceEngine
 * (zero-copy mmap, AccelTensor + Accelerate, no conversion).
 * </ul>
 *
 * <p>
 * Set {@code safetensor.provider.backend=direct} to use the new direct engine.
 */
@ApplicationScoped
public class SafetensorProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "safetensor";
    private static final Logger log = Logger.getLogger(SafetensorProvider.class);

    @Inject
    SafetensorProviderConfig config;
    @Inject
    DirectSafetensorBackend directBackend;
    @Inject
    AdapterMetricsRecorder adapterMetricsRecorder;

    @Inject
    ModelPipelineRegistry pipelineRegistry;

    @Inject
    Instance<SafetensorFeature> features;

    private final Map<String, SafetensorFeature> activeFeatures = new HashMap<>();

    // ── Provider identity ─────────────────────────────────────────────────────

    public boolean isEnabled() {
        return config.enabled();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "SafeTensor Provider (Direct)";
    }

    @Override
    public String version() {
        return "1.1.0";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(name())
                .version(version())
                .description("SafeTensor provider: native direct inference (zero-copy FFM)")
                .vendor("Golek / Kayys")
                .homepage("https://github.com/huggingface/safetensors")
                .build();
    }

    // ── Capabilities ──────────────────────────────────────────────────────────

    @Override
    public ProviderCapabilities capabilities() {
        var features = new LinkedHashSet<>(Set.of(
                "safetensors", "streaming-native", "direct-inference",
                "safetensors-native", "zero-copy-weights", "paged-kv-cache",
                "gqa", "rope", "swiglu", "lora-runtime", "speculative-decoding"));

        features.addAll(AdapterCapabilityProfile.builder()
                .adapterSupported(true)
                .adapterTypes(Set.of("lora"))
                .runtimeApply(true)
                .precompiledModelPath(false)
                .rolloutGuard(false)
                .metricsSchema(true)
                .build()
                .toFeatureFlags());

        return ProviderCapabilities.builder()
                .streaming(true)
                .embeddings(true)
                .multimodal(false)
                .functionCalling(false)
                .toolCalling(false)
                .structuredOutputs(false)
                .supportedFormats(Set.of(ModelFormat.SAFETENSORS))
                .supportedDevices(Set.of(tech.kayys.gollek.core.tensor.DeviceType.CPU))
                .features(Set.copyOf(features))
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ProviderConfig providerConfig)
            throws ProviderException.ProviderInitializationException {
        log.info("SafetensorProvider: initialising DIRECT backend");
        directBackend.initialize(providerConfig);
        initializeFeatures();
    }

    private void initializeFeatures() {
        var config = ConfigProvider.getConfig();
        features.stream()
                .sorted(Comparator.comparingInt(SafetensorFeature::priority))
                .forEach(feature -> {
                    String configKey = "gollek.safetensor.feature." + feature.id() + ".enabled";
                    boolean enabled = config.getOptionalValue(configKey, Boolean.class)
                            .orElse(feature.enabledByDefault());

                    if (enabled) {
                        try {
                            log.infof("Initialising Safetensor feature: %s", feature.id());
                            feature.initialize();
                            activeFeatures.put(feature.id(), feature);
                        } catch (Exception e) {
                            log.errorf(e, "Failed to initialise Safetensor feature: %s", feature.id());
                        }
                    } else {
                        log.debugf("Safetensor feature [%s] is disabled by configuration", feature.id());
                    }
                });
    }

    // ── Supports ──────────────────────────────────────────────────────────────

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (!config.enabled() || modelId == null || modelId.isBlank())
            return false;
        String modelRef = modelId;
        if (request != null) {
            modelRef = request.getParameter("model_path", String.class)
                    .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                    .orElse(modelId);
        }
        Path modelPath = resolveModelPath(modelRef);
        if (modelPath == null)
            return false;
        // Support directories (Hub-downloaded models) for DIRECT backend
        if (Files.isDirectory(modelPath)) {
            return Files.exists(modelPath.resolve("config.json"))
                    && (Files.exists(modelPath.resolve("tokenizer.json"))
                            || Files.exists(modelPath.resolve("model.safetensors.index.json")));
        }
        if (!Files.exists(modelPath))
            return false;
        String name = modelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return configuredExtensions().stream().anyMatch(name::endsWith);
    }

    // ── Infer ─────────────────────────────────────────────────────────────────

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        String modelRef = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());

        Path resolved = resolveModelPath(modelRef);
        if (resolved == null || (!Files.exists(resolved) && !Files.isDirectory(resolved))) {
            return Uni.createFrom().failure(new ProviderException(
                    "Model not found: " + friendlyModelName(modelRef)));
        }

        ModelPipelineRequest pipelineRequest = modelPipelineRequest(request, resolved);
        Optional<ModelPipeline> pipeline = selectFeaturePipeline(pipelineRequest);
        if (pipeline.isPresent()) {
            return pipeline.get().infer(pipelineRequest);
        }

        ProviderException validation = validateDirectModelCompatibility(resolved);
        if (validation != null) {
            return Uni.createFrom().failure(validation);
        }
        return directBackend.infer(request);
    }

    // ── InferStream ───────────────────────────────────────────────────────────

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        String modelRef = request.getParameter("model_path", String.class)
                .or(() -> Optional.ofNullable((String) request.getMetadata().get("model_path")))
                .orElse(request.getModel());

        Path resolved = resolveModelPath(modelRef);
        if (resolved == null || (!Files.exists(resolved) && !Files.isDirectory(resolved))) {
            return Multi.createFrom().failure(new ProviderException(
                    "Model not found: " + friendlyModelName(modelRef)));
        }

        ModelPipelineRequest pipelineRequest = modelPipelineRequest(request, resolved);
        Optional<ModelPipeline> pipeline = selectFeaturePipeline(pipelineRequest);
        if (pipeline.isPresent()) {
            return pipeline.get().inferStream(pipelineRequest);
        }

        ProviderException validation = validateDirectModelCompatibility(resolved);
        if (validation != null) {
            return Multi.createFrom().failure(validation);
        }
        return directBackend.inferStream(request);
    }

    // ── Health ────────────────────────────────────────────────────────────────

    @Override
    public Uni<ProviderHealth> health() {
        if (!config.enabled()) {
            return Uni.createFrom().item(ProviderHealth.builder()
                    .status(ProviderHealth.Status.UNHEALTHY)
                    .message("SafeTensor provider disabled")
                    .timestamp(Instant.now()).build());
        }
        return directBackend.health().map(h -> ProviderHealth.builder()
                .status(h.status()).message(h.message())
                .timestamp(Instant.now()).detail("delegate", "direct").build());
    }

    @Override
    public void shutdown() {
        /* no-op */ }

    // ── Resolvers ─────────────────────────────────────────────────────────────

    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank())
            return null;
        Path asGiven = Path.of(modelId);
        if (asGiven.isAbsolute()) {
            if (Files.exists(asGiven))
                return resolveToSafetensorFile(asGiven);
            for (String ext : configuredExtensions()) {
                Path withExt = Path.of(modelId + ext);
                if (Files.exists(withExt))
                    return resolveToSafetensorFile(withExt);
            }
            return resolveToSafetensorFile(asGiven);
        }
        Path fromBase = Path.of(config.basePath(), modelId);
        if (Files.exists(fromBase))
            return resolveToSafetensorFile(fromBase);
        for (String ext : configuredExtensions()) {
            Path withExt = Path.of(config.basePath(), modelId + ext);
            if (Files.exists(withExt))
                return resolveToSafetensorFile(withExt);
        }
        return resolveToSafetensorFile(Path.of(config.basePath(), modelId));
    }

    private Optional<ModelPipeline> selectFeaturePipeline(ModelPipelineRequest request) {
        if (pipelineRegistry == null) {
            return Optional.empty();
        }
        return pipelineRegistry.select(request);
    }

    private ProviderException validateDirectModelCompatibility(Path resolvedModelPath) {
        String modelName = friendlyModelName(resolvedModelPath == null ? null : resolvedModelPath.toString());
        Path checkpointDir = resolvedModelPath == null
                ? null
                : (Files.isDirectory(resolvedModelPath) ? resolvedModelPath : resolvedModelPath.getParent());
        Path configPath = checkpointDir == null ? null : checkpointDir.resolve("config.json");
        if (configPath == null || !Files.isRegularFile(configPath)) {
            return new ProviderException(
                    "Incomplete safetensor checkpoint (missing config.json). Re-pull model to download metadata sidecars.");
        }

        ModelConfig modelConfig;
        try {
            modelConfig = ModelConfig.load(configPath, new ObjectMapper());
        } catch (Exception e) {
            return new ProviderException(
                    "Invalid safetensor checkpoint config.json. Re-pull model to download metadata sidecars.", e);
        }

        if (ModelModalityTraits.detectVisionModel(modelConfig)) {
            return new ProviderException(
                    "Model config declares a vision/VLM architecture that is not supported by local safetensor text runtime: "
                            + modelName);
        }
        return null;
    }

    private ModelPipelineRequest modelPipelineRequest(ProviderRequest request, Path modelPath) {
        Map<String, Object> facts = new LinkedHashMap<>();
        put(facts, "provider", PROVIDER_ID);
        put(facts, "format", "safetensors");
        put(facts, "safetensor.backend", "direct");
        put(facts, "safetensor.model_path", modelPath == null ? null : modelPath.toAbsolutePath().normalize().toString());
        put(facts, "safetensor.model_name", modelPath == null || modelPath.getFileName() == null
                ? null
                : modelPath.getFileName().toString());
        put(facts, "safetensor.model_directory", modelPath == null ? null : Files.isDirectory(modelPath));
        put(facts, "safetensor.model_regular_file", modelPath == null ? null : Files.isRegularFile(modelPath));
        put(facts, "safetensor.model_size_bytes", regularFileSize(modelPath));
        Path modelDir = modelPath == null ? null : (Files.isDirectory(modelPath) ? modelPath : modelPath.getParent());
        put(facts, "safetensor.has_config_json", modelDir != null && Files.isRegularFile(modelDir.resolve("config.json")));
        put(facts, "safetensor.has_tokenizer_json", modelDir != null && Files.isRegularFile(modelDir.resolve("tokenizer.json")));
        put(facts, "safetensor.has_index_json", modelDir != null
                && Files.isRegularFile(modelDir.resolve("model.safetensors.index.json")));

        return ModelPipelineRequest.builder(request)
                .providerId(PROVIDER_ID)
                .modelPath(modelPath)
                .facts(facts)
                .build();
    }

    private static void put(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static Long regularFileSize(Path path) {
        try {
            return path != null && Files.isRegularFile(path) ? Files.size(path) : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private Path resolveToSafetensorFile(Path candidate) {
        if (candidate == null || !Files.exists(candidate))
            return candidate;
        if (!Files.isDirectory(candidate))
            return candidate;
        Path preferred = candidate.resolve("model.safetensors");
        if (Files.isRegularFile(preferred))
            return preferred;
        preferred = candidate.resolve("model.safetensor");
        if (Files.isRegularFile(preferred))
            return preferred;
        return candidate;
    }

    private List<String> configuredExtensions() {
        String raw = config.extensions();
        if (raw == null || raw.isBlank())
            return List.of(".safetensors", ".safetensor");
        return Arrays.stream(raw.split(",")).map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.startsWith(".") ? s.toLowerCase(Locale.ROOT) : "." + s.toLowerCase(Locale.ROOT))
                .toList();
    }

    private String friendlyModelName(String ref) {
        if (ref == null || ref.isBlank())
            return "unknown";
        try {
            Path p = Path.of(ref);
            Path fn = p.getFileName();
            String file = fn != null ? fn.toString() : ref;
            String lo = file.toLowerCase(Locale.ROOT);
            if (lo.endsWith(".safetensors") || lo.endsWith(".safetensor")) {
                Path par = p.getParent();
                if (par != null && par.getFileName() != null)
                    return par.getFileName().toString();
            }
            return file;
        } catch (Exception e) {
            return ref;
        }
    }
}
