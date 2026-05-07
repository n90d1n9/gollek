/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * DirectInferenceEngine.java
 * ──────────────────────────
 * Loads SafeTensor model weights using FFM mmap and bridges them to
 * AccelTensor — pure Java + Apple Accelerate. No LibTorch dependency.
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.models.core.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.function.Supplier;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;
import java.util.Map;
import java.util.Objects;
import java.util.Collection;
import java.util.Collections;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.Locale;

@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {
    private static final ThreadFactory STREAM_EXECUTOR_THREAD_FACTORY = runnable -> {
        Thread thread = new Thread(runnable, "gollek-direct-stream");
        thread.setDaemon(true);
        return thread;
    };
    private static final String PROFILE_PROPERTY = "gollek.profile";
    private static final ThreadLocal<InferenceProfile> ACTIVE_PROFILE = new ThreadLocal<>();

    @Inject
    Instance<tech.kayys.gollek.metal.MetalComputeBackend> metalBackend;

    /**
     * Direct SafeTensor inference engine using AccelTensor + Apple Accelerate.
     * No LibTorch dependency.
     */

    private static final Logger log = Logger.getLogger(DirectInferenceEngine.class);

    @Inject
    SafetensorLoaderFacade safetensorLoader;

    @Inject
    AccelWeightBridge bridge;

    @Inject
    QuantizationEngine quantizationEngine;

    @Inject
    ObjectMapper objectMapper;

    @Inject
    DirectForwardPass forwardPass;
    @Inject
    TokenSampler tokenSampler;
    @Inject
    ModelArchitectureRegistry archRegistry;

    @Inject
    KVCacheManager kvCacheManager;

    private final Map<Path, LoadedModel> modelsByPath = new ConcurrentHashMap<>();
    private final Map<String, LoadedModel> modelsByKey = new ConcurrentHashMap<>();
    private final Map<Path, Object> modelLocks = new ConcurrentHashMap<>();

    public static void recordAttentionNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) profile.attentionNanos += nanos;
    }

    public static void recordFfnNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) profile.ffnNanos += nanos;
    }

    public static void recordLogitsProjectionNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) profile.logitsProjectionNanos += nanos;
    }

    public static void recordLogitsMaterializationNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) profile.logitsMaterializationNanos += nanos;
    }

    private static boolean profilingEnabled() {
        return Boolean.getBoolean(PROFILE_PROPERTY);
    }

    private static InferenceProfile startProfile(String mode) {
        if (!profilingEnabled()) {
            return null;
        }
        InferenceProfile profile = new InferenceProfile(mode);
        ACTIVE_PROFILE.set(profile);
        return profile;
    }

    private static void clearProfile() {
        ACTIVE_PROFILE.remove();
    }

    private static String backendLabel(Instance<tech.kayys.gollek.metal.MetalComputeBackend> metalBackend) {
        if (metalBackend.isResolvable()) {
            String deviceName = metalBackend.get().deviceName();
            if (deviceName != null && !deviceName.contains("CPU")) {
                return "metal";
            }
        }
        return "cpu";
    }

    private static final class InferenceProfile {
        final String mode;
        long tokenizeNanos;
        long sessionAllocateNanos;
        long prefillNanos;
        long decodeNanos;
        long samplingNanos;
        long attentionNanos;
        long ffnNanos;
        long logitsProjectionNanos;
        long logitsMaterializationNanos;
        int decodeSteps;

        private InferenceProfile(String mode) {
            this.mode = mode;
        }

        Map<String, Object> metadata(String backend) {
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("profile_mode", mode);
            metadata.put("profile_backend", backend);
            metadata.put("profile_tokenize_ms", roundMillis(tokenizeNanos));
            metadata.put("profile_session_allocate_ms", roundMillis(sessionAllocateNanos));
            metadata.put("profile_prefill_ms", roundMillis(prefillNanos));
            metadata.put("profile_decode_ms", roundMillis(decodeNanos));
            metadata.put("profile_sampling_ms", roundMillis(samplingNanos));
            metadata.put("profile_attention_ms", roundMillis(attentionNanos));
            metadata.put("profile_ffn_ms", roundMillis(ffnNanos));
            metadata.put("profile_logits_ms", roundMillis(logitsProjectionNanos));
            metadata.put("profile_logits_materialization_ms", roundMillis(logitsMaterializationNanos));
            metadata.put("profile_decode_steps", decodeSteps);
            metadata.put("profile_summary", summary(backend));
            return metadata;
        }

        String summary(String backend) {
            return String.format(Locale.ROOT,
                    "backend=%s mode=%s tokenize=%.2fms session=%.2fms prefill=%.2fms decode=%.2fms sampling=%.2fms attention=%.2fms ffn=%.2fms logits=%.2fms logits_copy=%.2fms steps=%d",
                    backend, mode,
                    roundMillis(tokenizeNanos),
                    roundMillis(sessionAllocateNanos),
                    roundMillis(prefillNanos),
                    roundMillis(decodeNanos),
                    roundMillis(samplingNanos),
                    roundMillis(attentionNanos),
                    roundMillis(ffnNanos),
                    roundMillis(logitsProjectionNanos),
                    roundMillis(logitsMaterializationNanos),
                    decodeSteps);
        }

        private static double roundMillis(long nanos) {
            return nanos / 1_000_000.0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LoadedModel
    // ─────────────────────────────────────────────────────────────────────────

    public static class LoadedModel implements SafetensorEngine.LoadedModel {
        private final Path path;
        private final Tokenizer tokenizer;
        private final String key;
        private final boolean quantized;
        private final QuantizationEngine.QuantStrategy quantStrategy;
        private final String quantCacheState;
        private final Path quantCachePath;
        private final ModelConfig config;
        private final Map<String, AccelTensor> weights;
        private final Arena weightArena;

        public LoadedModel(Path path, Map<String, AccelTensor> weights,
                Tokenizer tokenizer, String key,
                boolean quantized, QuantizationEngine.QuantStrategy quantStrategy,
                String quantCacheState, Path quantCachePath,
                ModelConfig config, Arena weightArena) {
            this.path = path;
            this.weights = weights;
            this.tokenizer = tokenizer;
            this.key = key;
            this.quantized = quantized;
            this.quantStrategy = quantStrategy;
            this.quantCacheState = quantCacheState;
            this.quantCachePath = quantCachePath;
            this.config = config != null ? config : new ModelConfig();
            this.weightArena = weightArena;
        }

        @Override
        public Path path() {
            return path;
        }

        @Override
        public Map<String, AccelTensor> weights() {
            return weights;
        }

        @Override
        public Tokenizer tokenizer() {
            return tokenizer;
        }

        @Override
        public String key() {
            return key;
        }

        @Override
        public boolean isQuantized() {
            return quantized;
        }

        @Override
        public ModelConfig config() {
            return config;
        }

        public String arch() {
            return config.primaryArchitecture();
        }

        public QuantizationEngine.QuantStrategy getQuantStrategy() {
            return quantStrategy;
        }

        public String getQuantCacheState() {
            return quantCacheState;
        }

        public Path getQuantCachePath() {
            return quantCachePath;
        }
    }

    private record WeightLoadResult(Map<String, AccelTensor> weights, String quantCacheState, Path quantCachePath) {
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public String loadModel(Path modelPath) {
        return loadModel(modelPath, null, QuantizationEngine.QuantStrategy.NONE);
    }

    public String loadModel(Path modelPath, Path adapterPath,
            QuantizationEngine.QuantStrategy quantStrategy) {

        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        LoadedModel existing = modelsByPath.get(resolved);
        if (existing != null && existing.getQuantStrategy() == quantStrategy) {
            log.infof("DirectInferenceEngine: model already loaded [%s] (strategy=%s)",
                    resolved.getFileName(), quantStrategy);
            return existing.key();
        }

        synchronized (modelLocks.computeIfAbsent(resolved, k -> new Object())) {
            // Double-check inside lock
            existing = modelsByPath.get(resolved);
            if (existing != null && existing.getQuantStrategy() == quantStrategy) {
                return existing.key();
            }
            if (existing != null) {
                log.infof("DirectInferenceEngine: reloading [%s] for quantization strategy change %s -> %s",
                        resolved.getFileName(), existing.getQuantStrategy(), quantStrategy);
                unloadModel(resolved);
            }

            log.infof("DirectInferenceEngine: loading model [%s] (strategy=%s)",
                    resolved.getFileName(), quantStrategy);

            Arena weightArena = Arena.ofAuto();
            WeightLoadResult weightLoadResult = loadWeights(resolved, quantStrategy);
            Map<String, AccelTensor> weights = weightLoadResult.weights();

            Tokenizer tokenizer;
            try {
                tokenizer = loadTokenizer(resolved);
            } catch (Exception e) {
                log.warnf("Failed to load tokenizer for [%s]", resolved);
                throw new RuntimeException("Tokenizer loading failed", e);
            }

            ModelConfig config = loadConfig(resolved);

            String key = quantStrategy == QuantizationEngine.QuantStrategy.NONE
                    ? resolved.getFileName().toString()
                    : resolved.getFileName() + "#" + quantStrategy.name().toLowerCase();
            LoadedModel model = new LoadedModel(resolved, weights, tokenizer, key,
                    quantStrategy != QuantizationEngine.QuantStrategy.NONE, quantStrategy,
                    weightLoadResult.quantCacheState(), weightLoadResult.quantCachePath(),
                    config, weightArena);

            modelsByPath.put(resolved, model);
            modelsByKey.put(key, model);

            log.infof("DirectInferenceEngine: loaded [%s] — %d weights, arch=%s",
                    key, weights.size(), config.modelType());
            return key;
        }
    }

    public Uni<InferenceResponse> generate(String prompt, Path modelPath, GenerationConfig cfg) {
        return Uni.createFrom().item((Supplier<InferenceResponse>) () -> {
            Instant t0 = Instant.now();
            StringBuilder out = new StringBuilder();
            int inputLen = 0;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);

            try {
                if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null && !metalBackend.get().deviceName().contains("CPU")) {
                    System.out.println("Platform: Metal");
                    System.out.println("✓ GPU acceleration enabled (" + metalBackend.get().deviceName() + ")");
                } else {
                    System.out.println("Platform: Apple Silicon");
                    System.out.println("✓ CPU acceleration enabled (Accelerate AMX)");
                }
                System.out.flush();

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                if (verbose) { System.out.println("[DEBUG] 1: getLoadedModel"); System.out.flush(); }
                LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                if (model == null) {
                    if (verbose) { System.out.println("[DEBUG] 2: loadModel"); System.out.flush(); }
                    loadModel(modelPath);
                    model = (LoadedModel) getLoadedModel(modelPath);
                }

                if (model == null)
                    throw new RuntimeException("Model failed to load");

                if (verbose) { System.out.println("[DEBUG] 3: get tokenizer/config"); System.out.flush(); }
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                if (verbose) { System.out.println("[DEBUG] 4: arch resolve"); System.out.flush(); }
                ModelArchitecture arch = archRegistry.resolve(config);

                if (verbose) { System.out.println("[DEBUG] 5: tokenize"); System.out.flush(); }
                long tTokenize0 = System.nanoTime();
                long[] inputIds = tokenizer.encode(prompt, encodeOptionsFor(config));
                if (profile != null) profile.tokenizeNanos += System.nanoTime() - tTokenize0;
                inputLen = inputIds.length;
                if (inputLen == 0) {
                    throw new IllegalArgumentException("Prompt resulted in zero tokens. Please provide a valid prompt.");
                }
                if (verbose) { System.out.printf("[DEBUG] 6: tokens=%d\n", inputLen); System.out.flush(); }

                try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                    if (verbose) { System.out.println("[DEBUG] 7: allocate session"); System.out.flush(); }
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    if (profile != null) profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                    if (verbose) { System.out.println("[DEBUG] 8: prefill"); System.out.flush(); }
                    int[] freq = new int[config.vocabSize()];
                    Random rng = new Random();
                    boolean directGreedy = canUseDirectGreedySampling(cfg);

                    int next;
                    long tPrefill0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass.prefillLogitsTensor(inputIds, model.weights(), config, arch, session);
                        if (profile != null) profile.prefillNanos += System.nanoTime() - tPrefill0;
                        if (verbose) { System.out.println("[DEBUG] 9: prefill done"); System.out.flush(); }
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                        if (profile != null) profile.prefillNanos += System.nanoTime() - tPrefill0;
                        if (verbose) { System.out.println("[DEBUG] 9: prefill done"); System.out.flush(); }
                        applyLogitSoftcap(logits, config);
                        long tSample0 = System.nanoTime();
                        next = tokenSampler.sample(logits, cfg, config, freq, rng);
                        if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                    }

                    Set<Integer> stops = new HashSet<>();
                    for (int id : tokenizer.allStopTokenIds())
                        stops.add(id);
                    stops.addAll(config.eosTokenIds());
                    stops.addAll(cfg.stopTokenIds());

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        if (stops.contains(next))
                            break;

                        String delta = decoder.decodeNext((long) next);

                        out.append(delta);
                        String fullGeneratedText = decoder.currentText();
                        if (!cfg.stopStrings().isEmpty()) {
                            boolean shouldStop = false;
                            for (String s : cfg.stopStrings()) {
                                if (fullGeneratedText.endsWith(s)) {
                                    shouldStop = true;
                                    break;
                                }
                            }
                            if (shouldStop)
                                break;
                        }

                        if (next >= 0 && next < freq.length)
                            freq[next]++;
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        long tDecode0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass.decodeLogitsTensor(next, inputIds.length + step, model.weights(),
                                    config, arch, session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass.decode(next, inputIds.length + step, model.weights(), config, arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            applyLogitSoftcap(logits, config);
                            long tSample0 = System.nanoTime();
                            next = tokenSampler.sample(logits, cfg, config, freq, rng);
                            if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                    }
                }

                InferenceResponse.Builder builder = InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(out.length() / 4)
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor");
                if (profile != null) {
                    Map<String, Object> metadata = profile.metadata(backend);
                    metadata.forEach(builder::metadata);
                    System.out.println("[PROFILE] " + profile.summary(backend));
                    System.out.flush();
                }
                return builder.build();
            } catch (Exception e) {
                log.error("Generation failed", e);
                throw new RuntimeException("Direct generation failed: " + e.getMessage(), e);
            } finally {
                clearProfile();
            }
        });
    }

    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(STREAM_EXECUTOR_THREAD_FACTORY);
            executor.submit(() -> {
                Instant t0 = Instant.now();
                String requestId = UUID.randomUUID().toString();
                int inputLen = 0;
                InferenceProfile profile = startProfile("stream");
                String backend = backendLabel(metalBackend);

                try {
                    boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                    if (verbose) { System.out.println("[DEBUG-S] 1: getLoadedModel"); System.out.flush(); }
                    LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                    if (model == null) {
                        if (verbose) { System.out.println("[DEBUG-S] 2: loadModel"); System.out.flush(); }
                        loadModel(modelPath);
                        model = (LoadedModel) getLoadedModel(modelPath);
                    }

                    if (model == null)
                        throw new RuntimeException("Model failed to load");

                    if (verbose) { System.out.println("[DEBUG-S] 3: get tokenizer/config"); System.out.flush(); }
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    if (verbose) { System.out.println("[DEBUG-S] 4: arch resolve"); System.out.flush(); }
                    ModelArchitecture arch = archRegistry.resolve(config);

                    if (verbose) { System.out.println("[DEBUG-S] 5: tokenize"); System.out.flush(); }
                    long tTokenize0 = System.nanoTime();
                    long[] inputIds = tokenizer.encode(prompt, encodeOptionsFor(config));
                    if (profile != null) profile.tokenizeNanos += System.nanoTime() - tTokenize0;
                    inputLen = inputIds.length;
                    if (verbose) { System.out.printf("[DEBUG-S] 6: tokens=%d\n", inputLen); System.out.flush(); }

                    try (KVCacheManager.KVCacheSession session = kvCacheManager.createSession(cfg.maxKvCacheTokens())) {
                        if (verbose) { System.out.println("[DEBUG-S] 7: allocate session"); System.out.flush(); }
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        if (profile != null) profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                        if (verbose) { System.out.println("[DEBUG-S] 8: prefill"); System.out.flush(); }
                        int[] freq = new int[config.vocabSize()];
                        Random rng = new Random();
                        boolean directGreedy = canUseDirectGreedySampling(cfg);

                        int next;
                        long tPrefill0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass.prefillLogitsTensor(inputIds, model.weights(), config, arch, session);
                            if (profile != null) profile.prefillNanos += System.nanoTime() - tPrefill0;
                            if (verbose) { System.out.println("[DEBUG-S] 9: prefill done"); System.out.flush(); }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass.prefill(inputIds, model.weights(), config, arch, session);
                            if (profile != null) profile.prefillNanos += System.nanoTime() - tPrefill0;
                            if (verbose) { System.out.println("[DEBUG-S] 9: prefill done"); System.out.flush(); }
                            applyLogitSoftcap(logits, config);
                            long tSample0 = System.nanoTime();
                            next = tokenSampler.sample(logits, cfg, config, freq, rng);
                            if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                        }

                        Set<Integer> stops = new HashSet<>();
                        for (int id : tokenizer.allStopTokenIds())
                            stops.add(id);
                        stops.addAll(config.eosTokenIds());
                        stops.addAll(cfg.stopTokenIds());

                        StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                        for (int step = 0; step < cfg.maxNewTokens(); step++) {

                            if (stops.contains(next) || emitter.isCancelled()) {
                                break;
                            }

                            String delta = decoder.decodeNext((long) next);
                            if (delta == null)
                                delta = "";

                            String fullGeneratedText = decoder.currentText();

                            if (!delta.isEmpty()) {
                                emitter.emit(InferenceResponse.builder()
                                        .requestId(requestId)
                                        .content(delta)
                                        .model(modelPath.getFileName().toString())
                                        .inputTokens(inputLen)
                                        .metadata("backend", "accelerate-safetensor")
                                        .build());
                            }

                        if (!cfg.stopStrings().isEmpty()) {
                            boolean shouldStop = false;
                            for (String s : cfg.stopStrings()) {
                                    if (fullGeneratedText.endsWith(s)) {
                                        shouldStop = true;
                                        break;
                                    }
                                }
                                if (shouldStop)
                                    break;
                        }

                        if (next >= 0 && next < freq.length)
                            freq[next]++;
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        int decodeStartPos = inputIds.length + step;
                            long tDecode0 = System.nanoTime();
                            if (directGreedy) {
                                AccelTensor logits = forwardPass.decodeLogitsTensor(next, decodeStartPos, model.weights(), config, arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                long tSample0 = System.nanoTime();
                                next = sampleGreedyFromTensor(logits, config);
                                if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                            } else {
                                float[] logits = forwardPass.decode(next, decodeStartPos, model.weights(), config, arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                applyLogitSoftcap(logits, config);
                                long tSample0 = System.nanoTime();
                                next = tokenSampler.sample(logits, cfg, config, freq, rng);
                                if (profile != null) profile.samplingNanos += System.nanoTime() - tSample0;
                            }
                        }
                    }

                    InferenceResponse.Builder builder = InferenceResponse.builder()
                            .requestId(requestId)
                            .content("")
                            .model(modelPath.getFileName().toString())
                            .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                            .finishReason(InferenceResponse.FinishReason.STOP)
                            .inputTokens(inputLen)
                            .metadata("backend", "accelerate-safetensor");
                    if (profile != null) {
                        Map<String, Object> metadata = profile.metadata(backend);
                        metadata.forEach(builder::metadata);
                        System.out.println("[PROFILE] " + profile.summary(backend));
                        System.out.flush();
                    }
                    emitter.emit(builder.build());

                } catch (Throwable t) {
                    log.error("Generation failed", t);
                    emitter.fail(t);
                } finally {
                    clearProfile();
                    emitter.complete();
                    executor.shutdownNow();
                    try {
                        executor.awaitTermination(1, TimeUnit.SECONDS);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
        });
    }

    @Override
    public void unloadModel(Path modelPath) {
        Path resolved = modelPath.toAbsolutePath().normalize();
        LoadedModel model = modelsByPath.remove(resolved);
        if (model != null) {
            modelsByKey.remove(model.key());
            model.weights().values().forEach(t -> {
                try {
                    t.close();
                } catch (Exception ignored) {
                }
            });
            if (model.weightArena != null) {
                try {
                    model.weightArena.close();
                } catch (Exception ignored) {
                }
            }
            log.infof("DirectInferenceEngine: unloaded [%s]", resolved.getFileName());
        }
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(Path modelPath) {
        return modelsByPath.get(modelPath.toAbsolutePath().normalize());
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(String key) {
        return modelsByKey.get(key);
    }

    public boolean isLoaded(Path modelPath) {
        return modelsByPath.containsKey(modelPath.toAbsolutePath().normalize());
    }

    public Collection<LoadedModel> listLoadedModels() {
        return Collections.unmodifiableCollection(modelsByPath.values());
    }

    public QuantizationEngine getQuantizationEngine() {
        return quantizationEngine;
    }

    public KVCacheManager getKVCacheManager() {
        return kvCacheManager;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weight loading — pure AccelTensor, no LibTorch
    // ─────────────────────────────────────────────────────────────────────────

    private WeightLoadResult loadWeights(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy) {
        log.debugf("DirectInferenceEngine: opening weights from [%s]", modelPath);

        Map<String, AccelTensor> weights;
        try (SafetensorShardSession session = safetensorLoader.open(modelPath)) {
            weights = bridge.bridgeAll(session);
        } catch (Exception e) {
            log.errorf("Failed to load weights from %s: %s", modelPath, e.getMessage());
            throw new IllegalStateException("Failed to load weights: " + e.getMessage(), e);
        }

        String quantCacheState = "off";
        Path quantCachePath = null;
        if (quantStrategy != null && quantStrategy != QuantizationEngine.QuantStrategy.NONE) {
            weights = quantizationEngine.loadQuantizedModel(modelPath, quantStrategy);
            quantCacheState = "quantized-runtime";
        }

        applyWeightAliases(weights);
        log.infof("DirectInferenceEngine: bridged %d tensors (Accelerate/FFM backend, strategy=%s)",
                weights.size(), quantStrategy);
        return new WeightLoadResult(weights, quantCacheState, quantCachePath);
    }

    private void applyWeightAliases(Map<String, AccelTensor> weights) {
        List<Alias> aliases = List.of(
                new Alias("text_model.", "model."),
                new Alias("model.text_model.", "model."),
                new Alias("language_model.model.", "model."),
                new Alias("model.language_model.", "model."),
                new Alias("text_model.model.", "model."),
                new Alias("model.text_model.model.", "model."));

        Map<String, AccelTensor> updates = new HashMap<>();
        // Defensive copy of entries to prevent CME if another thread somehow
        // modifications the map
        List<Map.Entry<String, AccelTensor>> entries = new ArrayList<>(weights.entrySet());

        for (Alias alias : aliases) {
            for (Map.Entry<String, AccelTensor> entry : entries) {
                String key = entry.getKey();
                if (key.startsWith(alias.from())) {
                    String rewritten = alias.to() + key.substring(alias.from().length());
                    if (!weights.containsKey(rewritten)) {
                        updates.putIfAbsent(rewritten, entry.getValue());
                    }
                }
            }
        }
        weights.putAll(updates);
    }

    private record Alias(String from, String to) {
    }

    private Tokenizer loadTokenizer(Path modelPath) {
        try {
            return TokenizerFactory.load(modelPath, null);
        } catch (IOException e) {
            log.errorf(e, "Tokenizer loading failed for [%s]", modelPath);
            throw new RuntimeException("Tokenizer loading failed: " + e.getMessage(), e);
        }
    }

    private ModelConfig loadConfig(Path modelPath) {
        try {
            Path configDir = Files.isRegularFile(modelPath) ? modelPath.getParent() : modelPath;
            if (configDir != null) {
                return ModelConfig.fromDirectory(configDir, objectMapper);
            }
            return new ModelConfig();
        } catch (IOException e) {
            log.warnf(e, "DirectInferenceEngine: failed to read config.json near [%s]", modelPath);
            return new ModelConfig();
        }
    }

    private static void applyLogitSoftcap(float[] logits, ModelConfig config) {
        if (logits == null)
            return;
        Double cap = config.finalLogitSoftcapping();
        if (cap == null || cap <= 0)
            return;
        float c = cap.floatValue();
        for (int i = 0; i < logits.length; i++) {
            float x = logits[i];
            logits[i] = (float) (c * Math.tanh(x / c));
        }
    }

    private static boolean canUseDirectGreedySampling(GenerationConfig cfg) {
        return cfg.temperature() < 1.0e-4f
                && cfg.repetitionPenalty() == 1.0f
                && cfg.frequencyPenalty() == 0.0f;
    }

    private static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config) {
        try {
            java.lang.foreign.MemorySegment seg = logits.dataPtr();
            long vocab = logits.numel();
            Double cap = config.finalLogitSoftcapping();
            float softCap = cap != null && cap > 0 ? cap.floatValue() : 0.0f;

            int best = 0;
            float bestVal = applyLogitSoftcap(seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), softCap);
            for (int i = 1; i < vocab; i++) {
                float value = applyLogitSoftcap(seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i), softCap);
                if (value > bestVal) {
                    bestVal = value;
                    best = i;
                }
            }
            return best;
        } finally {
            logits.close();
        }
    }

    private static float applyLogitSoftcap(float value, float softCap) {
        if (softCap <= 0.0f) {
            return value;
        }
        return (float) (softCap * Math.tanh(value / softCap));
    }

    private static EncodeOptions encodeOptionsFor(ModelConfig config) {
        EncodeOptions options = EncodeOptions.defaultOptions();
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        if (modelType.startsWith("gemma")) {
            options.addBos = true;
        }
        return options;
    }
}
