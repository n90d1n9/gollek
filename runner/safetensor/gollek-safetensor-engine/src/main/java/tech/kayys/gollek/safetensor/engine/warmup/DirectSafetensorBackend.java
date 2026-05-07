/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.TextInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.models.core.ChatTemplateFormatter;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Flow;

/**
 * Direct Safetensor Backend for SafetensorProvider.
 * Delegates requests to the internal native inference engines.
 */
@ApplicationScoped
public class DirectSafetensorBackend {

    private static final Logger log = Logger.getLogger(DirectSafetensorBackend.class);
    private static final String PROVIDER_ID = "safetensor-direct";
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String VERBOSE_PROPERTY = "gollek.verbose";

    @Inject
    DirectInferenceEngine engine;

    @Inject
    TextInferenceEngine textEngine;

    @Inject
    tech.kayys.gollek.safetensor.audio.SpeechT5Engine speechT5Engine;

    @Inject
    SafetensorProviderConfig config;

    public void initialize(ProviderConfig providerConfig) {
        log.infof("DirectSafetensorBackend: initialised with FFM memory mapped weights");
    }

    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.builder()
                .status(ProviderHealth.Status.HEALTHY)
                .message("Direct Safetensor backend active (zero-copy enabled)")
                .timestamp(Instant.now())
                .build());
    }

    public Uni<InferenceResponse> infer(ProviderRequest request) {
        try {
            PreparedRequest prepared = prepareRequest(request);
            if (prepared.isAudioModel()) {
                long synthStart = System.currentTimeMillis();
                int promptTokens = prepared.prompt().length() / 4;
                java.util.Map<String, Object> metadata = new java.util.HashMap<>();
                appendQuantizationMetadata(metadata, prepared.loadedModel());
                metadata.put("audio", java.util.Base64.getEncoder().encodeToString(new byte[0]));
                return speechT5Engine.synthesize(prepared.ttsPrompt(), "alloy", prepared.modelPath(), prepared.audioConfig())
                        .map(bytes -> InferenceResponse.builder()
                                .requestId(request.getRequestId())
                                .model(request.getModel())
                                .content("")
                                .tokensUsed(promptTokens + (bytes.length / 1024))
                                .inputTokens(promptTokens)
                                .outputTokens(bytes.length / 1024)
                                .durationMs(System.currentTimeMillis() - synthStart)
                                .metadata(metadata)
                                .metadata("audio", java.util.Base64.getEncoder().encodeToString(bytes))
                                .metadata("modality", "AUDIO")
                                .build());
            }
            return textEngine.generate(prepared.prompt(), prepared.modelPath(), prepared.genCfg())
                    .map(infResp -> {
                        InferenceResponse.Builder builder = InferenceResponse.builder()
                            .requestId(infResp.getRequestId())
                            .model(request.getModel())
                            .content(infResp.getContent())
                            .tokensUsed(infResp.getTokensUsed())
                            .inputTokens(infResp.getInputTokens())
                            .outputTokens(infResp.getOutputTokens())
                            .durationMs(infResp.getDurationMs())
                            .finishReason(infResp.getFinishReason())
                            .metadata(infResp.getMetadata());
                        appendQuantizationMetadata(builder, prepared.loadedModel());
                        return builder.build();
                    });
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        java.util.concurrent.atomic.AtomicInteger idx = new java.util.concurrent.atomic.AtomicInteger(0);
        return Multi.createFrom().publisher(streamToPublisher(request))
                .map(response -> {
                    boolean isFinal = "stop".equals(response.getFinishReason());
                    int index = idx.getAndIncrement();
                    if ("AUDIO".equals(response.getMetadata().get("modality"))) {
                        return new StreamingInferenceChunk(response.getRequestId(), index, 
                                tech.kayys.gollek.spi.model.ModalityType.AUDIO, 
                                response.getContent(), null, isFinal, response.getFinishReason(), null, Instant.now(), response.getMetadata());
                    }
                    if (isFinal) {
                        return StreamingInferenceChunk.finalChunk(response.getRequestId(), index,
                                response.getContent());
                    } else {
                        return StreamingInferenceChunk.of(response.getRequestId(), index, response.getContent());
                    }
                });
    }

    private Flow.Publisher<ProviderResponse> streamToPublisher(ProviderRequest request) {
        try {
            PreparedRequest prepared = prepareRequest(request);
            if (prepared.isAudioModel()) {
                long synthStart = System.currentTimeMillis();
                int promptTokens = prepared.prompt().length() / 4;
                return Multi.createFrom().publisher(
                        speechT5Engine.synthesize(prepared.ttsPrompt(), "alloy", prepared.modelPath(), prepared.audioConfig())
                            .map(bytes -> {
                                long durationMs = System.currentTimeMillis() - synthStart;
                                String b64 = java.util.Base64.getEncoder().encodeToString(bytes);
                                return ProviderResponse.builder()
                                        .requestId(request.getRequestId())
                                        .model(request.getModel())
                                        .content(b64)
                                        .finishReason("stop")
                                        .promptTokens(promptTokens)
                                        .completionTokens(bytes.length / 1024) // rough "token" equivalent for audio bytes
                                        .metadata("modality", "AUDIO")
                                        .durationMs(durationMs)
                                        .build();
                            })
                            .toMulti()
                );
            }

            // Generate response lazily
            return textEngine.generateStream(prepared.prompt(), prepared.modelPath(), prepared.genCfg())
                    .map(infResp -> ProviderResponse.builder()
                            .requestId(infResp.getRequestId())
                            .content(infResp.getContent())
                            .model(infResp.getModel())
                            .finishReason(
                                    infResp.getFinishReason() != null ? infResp.getFinishReason().name().toLowerCase()
                                            : "stop")
                            .promptTokens(infResp.getInputTokens())
                            .completionTokens(infResp.getOutputTokens())
                            .totalTokens(infResp.getTokensUsed())
                            .durationMs(infResp.getDurationMs())
                            .build());
        } catch (Exception e) {
            String msg = (e.getMessage() != null) ? e.getMessage() : "(no message)";
            System.err.println("FATAL in streamToPublisher [" + e.getClass().getSimpleName() + "]: " + msg);
            e.printStackTrace(System.err);
            log.errorf(e, "Direct inference failed for model %s (resolved path: %s)",
                    request.getModel(), resolveModelPathFromRequest(request));
            throw new ProviderException("Direct inference failed for model "
                    + request.getModel() + " [" + e.getClass().getSimpleName() + "]: " + msg, e);
        }
    }

    private PreparedRequest prepareRequest(ProviderRequest request) {
        Path modelPath = resolveModelPathFromRequest(request);
        QuantizationEngine.QuantStrategy quantStrategy = parseQuantStrategy(request);
        engine.loadModel(modelPath, null, quantStrategy);

        var loadedModel = (DirectInferenceEngine.LoadedModel) engine.getLoadedModel(modelPath);
        String modelType = (loadedModel != null && loadedModel.config() != null)
                ? loadedModel.config().modelType()
                : "";
        String arch = (loadedModel != null && loadedModel.config() != null)
                ? loadedModel.config().primaryArchitecture()
                : "";

        boolean isAudioModel = ("speecht5".equals(modelType) || "whisper".equals(modelType)
                || modelPath.toString().contains("speecht5")
                || (arch != null && (arch.toLowerCase().contains("speecht5") || arch.toLowerCase().contains("whisper"))));
        boolean verbose = Boolean.getBoolean(VERBOSE_PROPERTY);

        configureForwardPlatform(modelType, arch);

        if (verbose) {
            System.err.println("[DEBUG-ROUTING] modelPath: " + modelPath);
            System.err.println("[DEBUG-ROUTING] loadedModel null? " + (loadedModel == null));
            System.err.println("[DEBUG-ROUTING] config null? " + (loadedModel != null && loadedModel.config() == null));
            System.err.println("[DEBUG-ROUTING] modelType: '" + modelType + "'");
            System.err.println("[DEBUG-ROUTING] arch: '" + arch + "'");
            System.err.println("[DEBUG-ROUTING] isAudioModel: " + isAudioModel);
        }

        List<tech.kayys.gollek.spi.Message> rawMessages = new ArrayList<>(request.getMessages());
        boolean hasSystem = rawMessages.stream()
                .anyMatch(m -> m.getRole() == tech.kayys.gollek.spi.Message.Role.SYSTEM);
        boolean injectDefaultSystem = !hasSystem
                && !requiresCpuForward(modelType, arch)
                && !shouldSkipDefaultSystemPrompt(modelType, arch);
        if (injectDefaultSystem) {
            String defaultSystem = modelType.toLowerCase().contains("qwen")
                    ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    : "You are a helpful assistant.";
            rawMessages.add(0, tech.kayys.gollek.spi.Message.system(defaultSystem));
        }

        @SuppressWarnings("unchecked")
        String prompt = ChatTemplateFormatter.format((java.util.List) (Object) rawMessages, modelType);
        if (verbose) {
            System.err.println("FORMATTED PROMPT: " + prompt.replace("\n", "\\n"));
        }

        int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : 256;
        int topK = request.getTopK();
        float topP = (float) request.getTopP();
        float temperature = (float) request.getTemperature();
        float minP = request.getParameter("min_p", Number.class)
                .map(Number::floatValue)
                .orElse(0.0f);
        long seed = request.getParameter("seed", Number.class)
                .map(Number::longValue)
                .orElse(-1L);
        String kvQuantStr = request.getParameter("kv_cache_quant", String.class).orElse("none");
        GenerationConfig.KvCacheQuantization kvQuant = normalizeKvQuantization(kvQuantStr);

        GenerationConfig genCfg = GenerationConfig.builder()
                .maxNewTokens(maxTokens)
                .strategy(resolveSamplingStrategy(temperature, topK, topP))
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .minP(minP)
                .repetitionPenalty((float) request.getRepeatPenalty())
                .kvCacheQuant(kvQuant)
                .seed(seed)
                .build();

        String ttsPrompt = rawMessages.stream()
                .filter(m -> m.getRole() == tech.kayys.gollek.spi.Message.Role.USER)
                .map(tech.kayys.gollek.spi.Message::getContent)
                .reduce((first, second) -> second)
                .orElse(prompt);

        String outputFormat = request.getParameter("output_format", String.class).orElse("wav");
        tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format format =
                tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format.WAV;
        try {
            format = tech.kayys.gollek.safetensor.audio.model.AudioConfig.Format.valueOf(outputFormat.toUpperCase());
        } catch (Exception ignored) {
        }

        tech.kayys.gollek.safetensor.audio.model.AudioConfig audioCfg =
                tech.kayys.gollek.safetensor.audio.model.AudioConfig.builder()
                        .temperature((float) request.getTemperature())
                        .format(format)
                        .build();

        return new PreparedRequest(modelPath, prompt, genCfg, isAudioModel, ttsPrompt, audioCfg, loadedModel);
    }

    private GenerationConfig.KvCacheQuantization normalizeKvQuantization(String raw) {
        if (raw == null || raw.isBlank() || "none".equalsIgnoreCase(raw)) {
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        String normalized = raw.trim().toUpperCase();
        if ("INT8".equals(normalized)) {
            return GenerationConfig.KvCacheQuantization.INT8;
        }
        if ("INT4".equals(normalized) || "TURBO".equals(normalized)) {
            log.warnf("KV cache quantization '%s' is not fully wired in the direct safetensor runtime yet; using NONE", raw);
            return GenerationConfig.KvCacheQuantization.NONE;
        }
        return GenerationConfig.KvCacheQuantization.NONE;
    }

    private GenerationConfig.SamplingStrategy resolveSamplingStrategy(float temperature, int topK, float topP) {
        if (temperature < 1.0e-4f || topK == 1) {
            return GenerationConfig.SamplingStrategy.GREEDY;
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0f && topP < 1.0f;
        if (hasTopK && hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_K_TOP_P;
        }
        if (hasTopP) {
            return GenerationConfig.SamplingStrategy.TOP_P;
        }
        if (hasTopK) {
            return GenerationConfig.SamplingStrategy.TOP_K;
        }
        return GenerationConfig.SamplingStrategy.GREEDY;
    }

    private void appendQuantizationMetadata(InferenceResponse.Builder builder, DirectInferenceEngine.LoadedModel loadedModel) {
        if (loadedModel == null || !loadedModel.isQuantized()) {
            return;
        }
        if (loadedModel.getQuantCacheState() != null) {
            builder.metadata("quant_cache_state", loadedModel.getQuantCacheState());
        }
        if (loadedModel.getQuantCachePath() != null) {
            builder.metadata("quant_cache_path", loadedModel.getQuantCachePath().toString());
        }
        builder.metadata("quant_strategy", loadedModel.getQuantStrategy().name().toLowerCase());
    }

    private void appendQuantizationMetadata(java.util.Map<String, Object> metadata, DirectInferenceEngine.LoadedModel loadedModel) {
        if (loadedModel == null || !loadedModel.isQuantized()) {
            return;
        }
        if (loadedModel.getQuantCacheState() != null) {
            metadata.put("quant_cache_state", loadedModel.getQuantCacheState());
        }
        if (loadedModel.getQuantCachePath() != null) {
            metadata.put("quant_cache_path", loadedModel.getQuantCachePath().toString());
        }
        metadata.put("quant_strategy", loadedModel.getQuantStrategy().name().toLowerCase());
    }

    private record PreparedRequest(
            Path modelPath,
            String prompt,
            GenerationConfig genCfg,
            boolean isAudioModel,
            String ttsPrompt,
            tech.kayys.gollek.safetensor.audio.model.AudioConfig audioConfig,
            DirectInferenceEngine.LoadedModel loadedModel) {
    }

    /**
     * Resolve the physical model path from a {@link ProviderRequest}.
     *
     * <p>Lookup order:
     * <ol>
     *   <li>{@code parameters["model_path"]} — set by ChatSessionManager for manifest-based models</li>
     *   <li>{@code metadata["model_path"]}   — injected by ModelRouterService.selectProvider()</li>
     *   <li>{@code request.getModel()}        — raw model ID / absolute path fallback</li>
     * </ol>
     */
    private Path resolveModelPathFromRequest(ProviderRequest request) {
        // 1. Check parameters first (set by ChatSessionManager)
        java.util.Optional<String> fromParam = request.getParameter("model_path", String.class);
        if (fromParam.isPresent() && !fromParam.get().isBlank()) {
            log.debugf("DirectSafetensorBackend: using model_path from parameters: %s", fromParam.get());
            return resolveModelPath(fromParam.get());
        }

        // 2. Check metadata (injected by ModelRouterService)
        Object fromMeta = request.getMetadata().get("model_path");
        if (fromMeta instanceof String s && !s.isBlank()) {
            log.debugf("DirectSafetensorBackend: using model_path from metadata: %s", s);
            return resolveModelPath(s);
        }

        // 3. Fallback to model ID (works for absolute paths or pre-configured aliases)
        log.debugf("DirectSafetensorBackend: no model_path override, resolving from model ID: %s", request.getModel());
        return resolveModelPath(request.getModel());
    }

    /**
     * Resolve a model path string (absolute path, or relative under basePath)
     * to a real filesystem path.
     */
    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            throw new ProviderException("Model ID is null or blank");
        }

        Path asPath = Path.of(modelId);

        // If the model ID is already an absolute path, use it directly
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        // Resolve relative to the configured base path
        Path resolved = Path.of(config.basePath(), modelId);
        if (Files.exists(resolved)) {
            log.debugf("Resolved model '%s' to path: %s", modelId, resolved);
            return resolved;
        }

        // Fall back to the as-given path (will likely fail but gives a clear error)
        log.warnf("Model path not found at %s, using as-is: %s", resolved, modelId);
        return asPath;
    }

    private QuantizationEngine.QuantStrategy parseQuantStrategy(ProviderRequest request) {
        String raw = request.getParameter("quantize_strategy", String.class).orElse("none");
        if (raw == null || raw.isBlank()) {
            return QuantizationEngine.QuantStrategy.NONE;
        }

        try {
            return QuantizationEngine.QuantStrategy.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            log.warnf("Unknown quantize_strategy '%s'; falling back to NONE", raw);
            return QuantizationEngine.QuantStrategy.NONE;
        }
    }

    private void configureForwardPlatform(String modelType, String arch) {
        if (Boolean.getBoolean("gollek.safetensor.allow_metal_gemma4")) {
            System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
            System.err.println("⚠ Safetensor forward path: allowing Metal for Gemma4 experimental validation");
            return;
        }
        if (requiresCpuForward(modelType, arch)) {
            System.setProperty(FORCE_CPU_FORWARD_PROPERTY, "true");
            System.err.println("⚠ Safetensor forward path: forcing CPU for " + arch + " to avoid incorrect Metal zero-logit output");
            return;
        }

        System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
    }

    private boolean requiresCpuForward(String modelType, String arch) {
        return false;
    }

    private boolean shouldSkipDefaultSystemPrompt(String modelType, String arch) {
        String mt = modelType == null ? "" : modelType.toLowerCase();
        String architecture = arch == null ? "" : arch.toLowerCase();
        return mt.startsWith("gemma4") || architecture.contains("gemma4");
    }

}
