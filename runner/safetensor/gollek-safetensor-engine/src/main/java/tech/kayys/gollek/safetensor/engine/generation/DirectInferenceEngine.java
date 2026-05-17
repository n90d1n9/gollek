/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ──────────────────────────
 * Loads SafeTensor model weights using FFM mmap and bridges them to
 * AccelTensor — pure Java + Apple Accelerate. No LibTorch dependency.
 */
package tech.kayys.gollek.safetensor.engine.generation;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.quarkus.arc.Arc;
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
import tech.kayys.gollek.safetensor.utils.SafetensorWriter;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.model.registry.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.metal.binding.MetalBinding;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.function.Supplier;
import java.util.function.Consumer;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Random;
import java.util.UUID;
import java.util.Map;
import java.util.Objects;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.Locale;
import java.util.Arrays;

@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {
    private static final ThreadFactory STREAM_EXECUTOR_THREAD_FACTORY = runnable -> {
        Thread thread = new Thread(runnable, "gollek-direct-stream");
        thread.setDaemon(true);
        return thread;
    };
    private static final String PROFILE_PROPERTY = "gollek.profile";
    private static final String DISABLE_METAL_F16_DISK_CACHE_PROPERTY =
            "gollek.safetensor.disable_metal_f16_disk_cache";
    private static final String METAL_F16_DISK_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_disk_cache_max_bytes";
    private static final String METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_disk_cache_min_free_bytes";
    private static final long GIB = 1024L * 1024L * 1024L;
    private static final long DEFAULT_METAL_F16_DISK_CACHE_MAX_BYTES = 12L * GIB;
    private static final long DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES = 2L * GIB;
    private static final long METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES = 256L * 1024L * 1024L;
    private static final ThreadLocal<InferenceProfile> ACTIVE_PROFILE = new ThreadLocal<>();

    @Inject
    Instance<Object> metalBackend;

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
        if (profile != null && profile.detailed)
            profile.attentionNanos += nanos;
    }

    public static void recordFfnNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed)
            profile.ffnNanos += nanos;
    }

    public static void recordLogitsProjectionNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed)
            profile.logitsProjectionNanos += nanos;
    }

    public static void recordLogitsMaterializationNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed)
            profile.logitsMaterializationNanos += nanos;
    }

    public static void recordLinearNanos(String operation, long nanos) {
        if (operation == null || operation.isBlank()) {
            return;
        }
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null && profile.detailed) {
            profile.linearNanosByOperation.merge(operation, nanos, Long::sum);
        }
    }

    private static boolean profilingEnabled() {
        return Boolean.getBoolean(PROFILE_PROPERTY);
    }

    /** Prints verbose timing breakdown only when {@code -Dgollek.profile=true}. */
    private static void maybePrintProfileSummary(InferenceProfile profile, String backend) {
        if (profile != null && profilingEnabled()) {
            System.out.println("[PROFILE] " + profile.summary(backend));
            System.out.flush();
        }
    }

    /**
     * Always allocate a lightweight profile so user-facing benchmark metadata is
     * populated without flags. Detailed operator breakdowns stay gated behind
     * {@code -Dgollek.profile=true} to avoid slowing the decode loop.
     */
    private static InferenceProfile startProfile(String mode) {
        InferenceProfile profile = new InferenceProfile(mode, profilingEnabled());
        ACTIVE_PROFILE.set(profile);
        return profile;
    }

    private static void clearProfile() {
        ACTIVE_PROFILE.remove();
    }

    private static void markFirstToken(InferenceProfile profile, long requestStartNanos) {
        if (profile != null && profile.firstTokenNanos <= 0L) {
            profile.firstTokenNanos = System.nanoTime() - requestStartNanos;
        }
    }

    private static void recordModelLoadNanos(long nanos) {
        InferenceProfile profile = ACTIVE_PROFILE.get();
        if (profile != null) {
            profile.modelLoadNanos += nanos;
        }
    }

    private static String backendLabel(Instance<?> metalBackend) {
        if (hasUsableMetalBackend(metalBackend) || isNativeMetalRuntimeActive()) {
            return "metal";
        }
        return "cpu";
    }

    private static boolean hasUsableMetalBackend(Instance<?> metalBackend) {
        if (metalBackend == null || !metalBackend.isResolvable()) {
            return false;
        }
        try {
            Object backend = metalBackend.get();
            String deviceName = (String) backend.getClass().getMethod("deviceName").invoke(backend);
            return deviceName != null && !deviceName.contains("CPU");
        } catch (Exception ignored) {
            return false;
        }
    }

    private static String metalDeviceLabel(Instance<?> metalBackend) {
        if (isNativeMetalRuntimeActive()) {
            try {
                return MetalBinding.getInstance().deviceName();
            } catch (Exception ignored) {
                // fall through
            }
        }
        if (metalBackend == null || !metalBackend.isResolvable()) {
            return null;
        }
        try {
            Object backend = metalBackend.get();
            return (String) backend.getClass().getMethod("deviceName").invoke(backend);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static boolean isNativeMetalRuntimeActive() {
        try {
            MetalBinding binding = MetalBinding.getInstance();
            return binding != null && binding.isNativeAvailable();
        } catch (Throwable ignored) {
            return false;
        }
    }

    private ModelArchitectureRegistry architectureRegistry() {
        if (archRegistry != null) {
            return archRegistry;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(ModelArchitectureRegistry.class);
                if (instance.isAvailable()) {
                    archRegistry = instance.get();
                }
            }
        } catch (Exception ignored) {
            // Fall through and raise a clearer error below.
        }
        if (archRegistry == null) {
            throw new IllegalStateException("ModelArchitectureRegistry is not available");
        }
        return archRegistry;
    }

    private DirectForwardPass forwardPass() {
        if (forwardPass != null) {
            return forwardPass;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(DirectForwardPass.class);
                if (instance.isAvailable()) {
                    forwardPass = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (forwardPass == null) {
            throw new IllegalStateException("DirectForwardPass is not available");
        }
        return forwardPass;
    }

    private TokenSampler tokenSampler() {
        if (tokenSampler != null) {
            return tokenSampler;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(TokenSampler.class);
                if (instance.isAvailable()) {
                    tokenSampler = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (tokenSampler == null) {
            throw new IllegalStateException("TokenSampler is not available");
        }
        return tokenSampler;
    }

    private SafetensorLoaderFacade safetensorLoader() {
        if (safetensorLoader != null) {
            return safetensorLoader;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(SafetensorLoaderFacade.class);
                if (instance.isAvailable()) {
                    safetensorLoader = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (safetensorLoader == null) {
            throw new IllegalStateException("SafetensorLoaderFacade is not available");
        }
        return safetensorLoader;
    }

    private AccelWeightBridge weightBridge() {
        if (bridge != null) {
            return bridge;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(AccelWeightBridge.class);
                if (instance.isAvailable()) {
                    bridge = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (bridge == null) {
            throw new IllegalStateException("AccelWeightBridge is not available");
        }
        return bridge;
    }

    private ObjectMapper resolvedObjectMapper() {
        if (objectMapper != null) {
            return objectMapper;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(ObjectMapper.class);
                if (instance.isAvailable()) {
                    objectMapper = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (objectMapper == null) {
            throw new IllegalStateException("ObjectMapper is not available");
        }
        return objectMapper;
    }

    private static final class InferenceProfile {
        final String mode;
        final boolean detailed;
        long tokenizeNanos;
        long modelLoadNanos;
        long sessionAllocateNanos;
        long firstTokenNanos;
        long prefillNanos;
        long decodeNanos;
        long samplingNanos;
        long attentionNanos;
        long ffnNanos;
        long logitsProjectionNanos;
        long logitsMaterializationNanos;
        final Map<String, Long> linearNanosByOperation = new LinkedHashMap<>();
        int decodeSteps;

        private InferenceProfile(String mode, boolean detailed) {
            this.mode = mode;
            this.detailed = detailed;
        }

        Map<String, Object> metadata(String backend) {
            return metadata(backend, -1, -1);
        }

        /**
         * @param promptTokens     prompt token count for llama.cpp-style bench keys;
         *                         {@code -1} skips bench/tokens
         * @param completionTokens generated token count (excluding EOS stop)
         */
        Map<String, Object> metadata(String backend, int promptTokens, int completionTokens) {
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("profile_mode", mode);
            metadata.put("profile_backend", backend);
            metadata.put("profile_load_ms", roundMillis(modelLoadNanos));
            if (modelLoadNanos > 0L) {
                metadata.put("bench.load_ms", roundMillis(modelLoadNanos));
            }
            metadata.put("profile_tokenize_ms", roundMillis(tokenizeNanos));
            metadata.put("profile_session_allocate_ms", roundMillis(sessionAllocateNanos));
            if (firstTokenNanos > 0L) {
                metadata.put("profile_ttft_ms", roundMillis(firstTokenNanos));
                metadata.put("bench.ttft_ms", roundMillis(firstTokenNanos));
            }
            metadata.put("profile_prefill_ms", roundMillis(prefillNanos));
            metadata.put("profile_decode_ms", roundMillis(decodeNanos));
            metadata.put("profile_sampling_ms", roundMillis(samplingNanos));
            metadata.put("profile_attention_ms", roundMillis(attentionNanos));
            metadata.put("profile_ffn_ms", roundMillis(ffnNanos));
            metadata.put("profile_logits_ms", roundMillis(logitsProjectionNanos));
            metadata.put("profile_logits_materialization_ms", roundMillis(logitsMaterializationNanos));
            linearNanosByOperation.forEach((operation, nanos) -> metadata
                    .put("profile_linear_" + sanitizeMetricKey(operation) + "_ms", roundMillis(nanos)));
            metadata.put("profile_decode_steps", decodeSteps);
            metadata.put("profile_summary", summary(backend));
            if (promptTokens >= 0) {
                metadata.put("tokens.input", promptTokens);
                metadata.put("tokens.output", Math.max(0, completionTokens));
                double prefillSec = prefillNanos / 1_000_000_000.0;
                double decodeSec = decodeNanos / 1_000_000_000.0;
                if (promptTokens > 0 && prefillSec > 1e-12) {
                    metadata.put("bench.prefill_tps", promptTokens / prefillSec);
                }
                if (completionTokens > 0 && decodeSec > 1e-12) {
                    metadata.put("bench.generation_tps", completionTokens / decodeSec);
                }
                if (decodeSteps > 0 && decodeNanos > 0L) {
                    metadata.put("bench.tpot_ms", roundMillis(decodeNanos) / decodeSteps);
                }
            }
            return metadata;
        }

        String summary(String backend) {
            return String.format(Locale.ROOT,
                    "backend=%s mode=%s load=%.2fms tokenize=%.2fms session=%.2fms ttft=%.2fms prefill=%.2fms decode=%.2fms tpot=%.2fms sampling=%.2fms attention=%.2fms ffn=%.2fms logits=%.2fms logits_copy=%.2fms steps=%d%s",
                    backend, mode,
                    roundMillis(modelLoadNanos),
                    roundMillis(tokenizeNanos),
                    roundMillis(sessionAllocateNanos),
                    roundMillis(firstTokenNanos),
                    roundMillis(prefillNanos),
                    roundMillis(decodeNanos),
                    decodeSteps > 0 ? roundMillis(decodeNanos) / decodeSteps : 0.0,
                    roundMillis(samplingNanos),
                    roundMillis(attentionNanos),
                    roundMillis(ffnNanos),
                    roundMillis(logitsProjectionNanos),
                    roundMillis(logitsMaterializationNanos),
                    decodeSteps,
                    linearSummarySuffix());
        }

        private static double roundMillis(long nanos) {
            return nanos / 1_000_000.0;
        }

        private String linearSummarySuffix() {
            if (linearNanosByOperation.isEmpty()) {
                return "";
            }
            StringBuilder sb = new StringBuilder(" linear={");
            linearNanosByOperation.entrySet().stream()
                    .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder()))
                    .limit(8)
                    .forEachOrdered(entry -> {
                        if (sb.length() > " linear={".length()) {
                            sb.append(", ");
                        }
                        sb.append(entry.getKey())
                                .append('=')
                                .append(String.format(Locale.ROOT, "%.2f", roundMillis(entry.getValue())));
                    });
            sb.append('}');
            return sb.toString();
        }

        private static String sanitizeMetricKey(String raw) {
            StringBuilder out = new StringBuilder(raw.length());
            for (int i = 0; i < raw.length(); i++) {
                char ch = raw.charAt(i);
                if ((ch >= 'a' && ch <= 'z')
                        || (ch >= 'A' && ch <= 'Z')
                        || (ch >= '0' && ch <= '9')) {
                    out.append(Character.toLowerCase(ch));
                } else {
                    out.append('_');
                }
            }
            return out.toString();
        }
    }

    public record DirectGenerationTrace(
            InferenceResponse response,
            long[] inputIds,
            long[] generatedTokenIds) {
    }

    public record DirectConversationTrace(
            InferenceResponse response,
            long[] inputIds,
            long[] generatedTokenIds,
            KVCacheManager.KVCacheSession kvCacheSession) {
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
            long requestStartNanos = System.nanoTime();
            StringBuilder out = new StringBuilder();
            int inputLen = 0;
            int completionTokens = 0;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);

            try {
                String metalDevice = metalDeviceLabel(metalBackend);
                if (metalDevice != null && !metalDevice.contains("CPU")) {
                    System.out.println("Platform: Metal");
                    System.out.println("✓ GPU acceleration enabled (" + metalDevice + ")");
                } else {
                    System.out.println("Platform: Apple Silicon");
                    System.out.println("✓ CPU acceleration enabled (Accelerate AMX)");
                }
                System.out.flush();

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                if (verbose) {
                    System.out.println("[DEBUG] 1: getLoadedModel");
                    System.out.flush();
                }
                LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                if (model == null) {
                    if (verbose) {
                        System.out.println("[DEBUG] 2: loadModel");
                        System.out.flush();
                    }
                    long tLoad0 = System.nanoTime();
                    loadModel(modelPath);
                    recordModelLoadNanos(System.nanoTime() - tLoad0);
                    model = (LoadedModel) getLoadedModel(modelPath);
                }

                if (model == null)
                    throw new RuntimeException("Model failed to load");

                if (verbose) {
                    System.out.println("[DEBUG] 3: get tokenizer/config");
                    System.out.flush();
                }
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                if (verbose) {
                    System.out.println("[DEBUG] 4: arch resolve");
                    System.out.flush();
                }
                ModelArchitecture arch = architectureRegistry().resolve(config);

                if (verbose) {
                    System.out.println("[DEBUG] 5: tokenize");
                    System.out.flush();
                }
                long tTokenize0 = System.nanoTime();
                long[] inputIds = tokenizer.encode(prompt, encodeOptionsFor(config, prompt));
                if (profile != null)
                    profile.tokenizeNanos += System.nanoTime() - tTokenize0;
                inputLen = inputIds.length;
                if (inputLen == 0) {
                    throw new IllegalArgumentException(
                            "Prompt resulted in zero tokens. Please provide a valid prompt.");
                }
                if (verbose) {
                    System.out.printf("[DEBUG] 6: tokens=%d\n", inputLen);
                    System.out.flush();
                }
                if (verbose) {
                    debugTokenSequence(tokenizer, inputIds, "prompt");
                }

                Set<Integer> stops = new HashSet<>();
                for (int id : tokenizer.allStopTokenIds())
                    stops.add(id);
                stops.addAll(config.eosTokenIds());
                stops.addAll(cfg.stopTokenIds());

                try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                        .createSession(cfg.maxKvCacheTokens())) {
                    if (verbose) {
                        System.out.println("[DEBUG] 7: allocate session");
                        System.out.flush();
                    }
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    if (profile != null)
                        profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                    if (verbose) {
                        System.out.println("[DEBUG] 8: prefill");
                        System.out.flush();
                    }
                    boolean directGreedy = canUseDirectGreedySampling(cfg);
                    int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                    Random rng = directGreedy ? null : new Random();
                    GreedySamplingMasks greedyMasks = directGreedy
                            ? buildGreedySamplingMasks(tokenizer, stops, config.vocabSize())
                            : null;

                    int next;
                    long tPrefill0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config, arch,
                                session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        if (verbose) {
                            System.out.println("[DEBUG] 9: prefill done");
                            System.out.flush();
                        }
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config, tokenizer, true, stops, greedyMasks);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        if (verbose) {
                            System.out.println("[DEBUG] 9: prefill done");
                            System.out.flush();
                        }
                        long tSample0 = System.nanoTime();
                        next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, true, stops);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    }
                    if (verbose) {
                        debugChosenToken(tokenizer, next, 0);
                    }

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        if (stops.contains(next))
                            break;

                        markFirstToken(profile, requestStartNanos);
                        String delta = decoder.decodeNext((long) next);

                        out.append(delta);
                        completionTokens++;
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

                        recordSampleFrequency(freq, next);
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        long tDecode0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().decodeLogitsTensor(next, inputIds.length + step,
                                    model.weights(),
                                    config, arch, session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config, tokenizer, false, stops, greedyMasks);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().decode(next, inputIds.length + step, model.weights(), config,
                                    arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, false, stops);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                        if (verbose) {
                            debugChosenToken(tokenizer, next, step + 1);
                        }
                    }
                }

                InferenceResponse.Builder builder = InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(completionTokens)
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor");
                if (profile != null) {
                    Map<String, Object> metadata = profile.metadata(backend, inputLen, completionTokens);
                    metadata.forEach(builder::metadata);
                    maybePrintProfileSummary(profile, backend);
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

    public long[] encodePrompt(String prompt, Path modelPath) {
        LoadedModel model = requireLoadedModel(modelPath);
        return model.tokenizer().encode(prompt, encodeOptionsFor(model.config(), prompt));
    }

    public Uni<DirectGenerationTrace> generateWithTrace(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return Uni.createFrom().item((Supplier<DirectGenerationTrace>) () -> {
            Instant t0 = Instant.now();
            long requestStartNanos = System.nanoTime();
            StringBuilder out = new StringBuilder();
            int inputLen = 0;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);
            List<Long> generatedTokenIds = new ArrayList<>();

            try {
                String metalDevice = metalDeviceLabel(metalBackend);
                if (metalDevice != null && !metalDevice.contains("CPU")) {
                    System.out.println("Platform: Metal");
                    System.out.println("✓ GPU acceleration enabled (" + metalDevice + ")");
                } else {
                    System.out.println("Platform: Apple Silicon");
                    System.out.println("✓ CPU acceleration enabled (Accelerate AMX)");
                }
                System.out.flush();

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                LoadedModel model = requireLoadedModel(modelPath);
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = architectureRegistry().resolve(config);

                inputLen = inputIds.length;
                if (inputLen == 0) {
                    throw new IllegalArgumentException(
                            "Prompt resulted in zero tokens. Please provide a valid prompt.");
                }
                if (verbose) {
                    System.out.printf("[DEBUG] pretokenized tokens=%d%n", inputLen);
                    System.out.flush();
                }

                try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                        .createSession(cfg.maxKvCacheTokens())) {
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    if (profile != null)
                        profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                    boolean directGreedy = canUseDirectGreedySampling(cfg);
                    int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                    Random rng = directGreedy ? null : new Random();

                    int next;
                    long tPrefill0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config, arch,
                                session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = tokenSampler().sample(logits, cfg, config, freq, rng);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
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

                        markFirstToken(profile, requestStartNanos);
                        generatedTokenIds.add((long) next);

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

                        recordSampleFrequency(freq, next);
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        long tDecode0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().decodeLogitsTensor(next, inputIds.length + step,
                                    model.weights(),
                                    config, arch, session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().decode(next, inputIds.length + step, model.weights(), config,
                                    arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = tokenSampler().sample(logits, cfg, config, freq, rng);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                    }
                }

                InferenceResponse.Builder builder = InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(generatedTokenIds.size())
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor")
                        .metadata("prompt_token_source", "pretokenized");
                if (profile != null) {
                    Map<String, Object> metadata = profile.metadata(backend, inputLen, generatedTokenIds.size());
                    metadata.forEach(builder::metadata);
                    maybePrintProfileSummary(profile, backend);
                }
                return new DirectGenerationTrace(
                        builder.build(),
                        inputIds.clone(),
                        toLongArray(generatedTokenIds));
            } catch (Exception e) {
                log.error("Generation failed", e);
                throw new RuntimeException("Direct generation failed: " + e.getMessage(), e);
            } finally {
                clearProfile();
            }
        });
    }

    public Uni<InferenceResponse> generate(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return generateWithTrace(inputIds, modelPath, cfg)
                .map(DirectGenerationTrace::response);
    }

    public Uni<DirectConversationTrace> generateWithConversationTrace(long[] inputIds, Path modelPath,
            GenerationConfig cfg) {
        return Uni.createFrom().item((Supplier<DirectConversationTrace>) () -> {
            Instant t0 = Instant.now();
            long requestStartNanos = System.nanoTime();
            StringBuilder out = new StringBuilder();
            int inputLen = 0;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);
            List<Long> generatedTokenIds = new ArrayList<>();
            KVCacheManager.KVCacheSession session = null;

            try {
                String metalDevice = metalDeviceLabel(metalBackend);
                if (metalDevice != null && !metalDevice.contains("CPU")) {
                    System.out.println("Platform: Metal");
                    System.out.println("✓ GPU acceleration enabled (" + metalDevice + ")");
                } else {
                    System.out.println("Platform: Apple Silicon");
                    System.out.println("✓ CPU acceleration enabled (Accelerate AMX)");
                }
                System.out.flush();

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                LoadedModel model = requireLoadedModel(modelPath);
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = architectureRegistry().resolve(config);

                inputLen = inputIds.length;
                if (inputLen == 0) {
                    throw new IllegalArgumentException(
                            "Prompt resulted in zero tokens. Please provide a valid prompt.");
                }
                if (verbose) {
                    System.out.printf("[DEBUG] conversational pretokenized tokens=%d%n", inputLen);
                    System.out.flush();
                }

                session = getKVCacheManager().createSession(cfg.maxKvCacheTokens());
                long tAlloc0 = System.nanoTime();
                session.allocate(config, cfg);
                if (profile != null)
                    profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                boolean directGreedy = canUseDirectGreedySampling(cfg);
                int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                Random rng = directGreedy ? null : new Random();

                int next;
                long tPrefill0 = System.nanoTime();
                if (directGreedy) {
                    AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config, arch,
                            session);
                    if (profile != null)
                        profile.prefillNanos += System.nanoTime() - tPrefill0;
                    long tSample0 = System.nanoTime();
                    next = sampleGreedyFromTensor(logits, config);
                    if (profile != null)
                        profile.samplingNanos += System.nanoTime() - tSample0;
                } else {
                    float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                    if (profile != null)
                        profile.prefillNanos += System.nanoTime() - tPrefill0;
                    long tSample0 = System.nanoTime();
                    next = tokenSampler().sample(logits, cfg, config, freq, rng);
                    if (profile != null)
                        profile.samplingNanos += System.nanoTime() - tSample0;
                }

                Set<Integer> stops = new HashSet<>();
                for (int id : tokenizer.allStopTokenIds()) {
                    stops.add(id);
                }
                stops.addAll(config.eosTokenIds());
                stops.addAll(cfg.stopTokenIds());

                StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                for (int step = 0; step < cfg.maxNewTokens(); step++) {
                    if (stops.contains(next)) {
                        break;
                    }

                    markFirstToken(profile, requestStartNanos);
                    generatedTokenIds.add((long) next);
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
                        if (shouldStop) {
                            break;
                        }
                    }

                    recordSampleFrequency(freq, next);
                    if (step + 1 >= cfg.maxNewTokens()) {
                        break;
                    }
                    long tDecode0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().decodeLogitsTensor(next, inputIds.length + step,
                                model.weights(),
                                config, arch, session);
                        if (profile != null) {
                            profile.decodeNanos += System.nanoTime() - tDecode0;
                            profile.decodeSteps++;
                        }
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().decode(next, inputIds.length + step, model.weights(), config,
                                arch,
                                session);
                        if (profile != null) {
                            profile.decodeNanos += System.nanoTime() - tDecode0;
                            profile.decodeSteps++;
                        }
                        long tSample0 = System.nanoTime();
                        next = tokenSampler().sample(logits, cfg, config, freq, rng);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    }
                }

                InferenceResponse.Builder builder = InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(generatedTokenIds.size())
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor")
                        .metadata("prompt_token_source", "pretokenized")
                        .metadata("conversation_kv_retained", true);
                if (profile != null) {
                    Map<String, Object> metadata = profile.metadata(backend, inputLen, generatedTokenIds.size());
                    metadata.forEach(builder::metadata);
                    maybePrintProfileSummary(profile, backend);
                }
                KVCacheManager.KVCacheSession retained = session;
                session = null;
                return new DirectConversationTrace(
                        builder.build(),
                        inputIds.clone(),
                        toLongArray(generatedTokenIds),
                        retained);
            } catch (Exception e) {
                if (session != null) {
                    try {
                        session.close();
                    } catch (Exception ignored) {
                    }
                }
                log.error("Conversation generation failed", e);
                throw new RuntimeException("Direct conversation generation failed: " + e.getMessage(), e);
            } finally {
                clearProfile();
            }
        });
    }

    public Uni<DirectConversationTrace> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg) {
        return generateContinuationWithConversationTrace(
                fullInputIds,
                cachedPrefixTokens,
                session,
                modelPath,
                cfg,
                null);
    }

    public Uni<DirectConversationTrace> generateContinuationWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Integer replayTokenId) {
        return Uni.createFrom().item((Supplier<DirectConversationTrace>) () -> {
            Instant t0 = Instant.now();
            long requestStartNanos = System.nanoTime();
            StringBuilder out = new StringBuilder();
            int inputLen = fullInputIds.length;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);
            List<Long> generatedTokenIds = new ArrayList<>();

            try {
                String metalDevice = metalDeviceLabel(metalBackend);
                if (metalDevice != null && !metalDevice.contains("CPU")) {
                    System.out.println("Platform: Metal");
                    System.out.println("✓ GPU acceleration enabled (" + metalDevice + ")");
                } else {
                    System.out.println("Platform: Apple Silicon");
                    System.out.println("✓ CPU acceleration enabled (Accelerate AMX)");
                }
                System.out.flush();

                if (session == null) {
                    throw new IllegalArgumentException("Conversation continuation requires an active KV cache session");
                }
                if (cachedPrefixTokens < 0 || cachedPrefixTokens > fullInputIds.length) {
                    throw new IllegalArgumentException("Invalid cachedPrefixTokens: " + cachedPrefixTokens);
                }

                LoadedModel model = requireLoadedModel(modelPath);
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                ModelArchitecture arch = architectureRegistry().resolve(config);

                long[] deltaInputIds = java.util.Arrays.copyOfRange(fullInputIds, cachedPrefixTokens,
                        fullInputIds.length);
                boolean exactReplay = deltaInputIds.length == 0;
                if (exactReplay && replayTokenId == null) {
                    throw new IllegalArgumentException(
                            "Conversation continuation requires at least one delta token or a replay token");
                }
                if (session.currentPos() != cachedPrefixTokens) {
                    throw new IllegalStateException("KV cache position " + session.currentPos()
                            + " does not match cached prefix tokens " + cachedPrefixTokens);
                }

                boolean directGreedy = canUseDirectGreedySampling(cfg);
                int[] freq = directGreedy ? null : initializePromptFrequencies(config, fullInputIds);
                Random rng = directGreedy ? null : new Random();

                int next;
                if (exactReplay) {
                    next = replayTokenId;
                } else {
                    long tPrefill0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().prefillLogitsTensor(deltaInputIds, model.weights(), config,
                                arch, session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().prefill(deltaInputIds, model.weights(), config, arch, session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = tokenSampler().sample(logits, cfg, config, freq, rng);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    }
                }

                Set<Integer> stops = new HashSet<>();
                for (int id : tokenizer.allStopTokenIds()) {
                    stops.add(id);
                }
                stops.addAll(config.eosTokenIds());
                stops.addAll(cfg.stopTokenIds());

                StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                for (int step = 0; step < cfg.maxNewTokens(); step++) {
                    if (stops.contains(next)) {
                        break;
                    }

                    markFirstToken(profile, requestStartNanos);
                    generatedTokenIds.add((long) next);
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
                        if (shouldStop) {
                            break;
                        }
                    }

                    recordSampleFrequency(freq, next);
                    if (step + 1 >= cfg.maxNewTokens()) {
                        break;
                    }
                    long tDecode0 = System.nanoTime();
                    int decodeStartPos = fullInputIds.length + step;
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().decodeLogitsTensor(next, decodeStartPos, model.weights(),
                                config, arch, session);
                        if (profile != null) {
                            profile.decodeNanos += System.nanoTime() - tDecode0;
                            profile.decodeSteps++;
                        }
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().decode(next, decodeStartPos, model.weights(), config, arch,
                                session);
                        if (profile != null) {
                            profile.decodeNanos += System.nanoTime() - tDecode0;
                            profile.decodeSteps++;
                        }
                        long tSample0 = System.nanoTime();
                        next = tokenSampler().sample(logits, cfg, config, freq, rng);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    }
                }

                InferenceResponse.Builder builder = InferenceResponse.builder()
                        .requestId(UUID.randomUUID().toString())
                        .content(out.toString())
                        .model(modelPath.getFileName().toString())
                        .inputTokens(inputLen)
                        .outputTokens(generatedTokenIds.size())
                        .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .metadata("backend", "accelerate-safetensor")
                        .metadata("prompt_token_source", exactReplay ? "conversation_replay" : "conversation_delta")
                        .metadata("conversation_delta_prefill", !exactReplay)
                        .metadata("conversation_exact_replay", exactReplay)
                        .metadata("conversation_cached_prefix_tokens", cachedPrefixTokens)
                        .metadata("conversation_delta_prompt_tokens", deltaInputIds.length)
                        .metadata("conversation_kv_retained", true);
                if (profile != null) {
                    Map<String, Object> metadata = profile.metadata(backend, inputLen, generatedTokenIds.size());
                    metadata.forEach(builder::metadata);
                    maybePrintProfileSummary(profile, backend);
                }
                return new DirectConversationTrace(
                        builder.build(),
                        fullInputIds.clone(),
                        toLongArray(generatedTokenIds),
                        session);
            } catch (Exception e) {
                log.error("Conversation continuation failed", e);
                throw new RuntimeException("Direct conversation continuation failed: " + e.getMessage(), e);
            } finally {
                clearProfile();
            }
        });
    }

    public Multi<InferenceResponse> generateStreamWithConversationTrace(
            long[] inputIds,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete) {
        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(STREAM_EXECUTOR_THREAD_FACTORY);
            executor.submit(() -> {
                Instant t0 = Instant.now();
                long requestStartNanos = System.nanoTime();
                String requestId = UUID.randomUUID().toString();
                int inputLen = inputIds.length;
                InferenceProfile profile = startProfile("stream");
                String backend = backendLabel(metalBackend);
                List<Long> generatedTokenIds = new ArrayList<>();
                KVCacheManager.KVCacheSession session = null;

                try {
                    LoadedModel model = requireLoadedModel(modelPath);
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    ModelArchitecture arch = architectureRegistry().resolve(config);

                    session = getKVCacheManager().createSession(cfg.maxKvCacheTokens());
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    if (profile != null)
                        profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                    boolean directGreedy = canUseDirectGreedySampling(cfg);
                    int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                    Random rng = directGreedy ? null : new Random();

                    int next;
                    long tPrefill0 = System.nanoTime();
                    if (directGreedy) {
                        AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config, arch,
                                session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = sampleGreedyFromTensor(logits, config);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    } else {
                        float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                        if (profile != null)
                            profile.prefillNanos += System.nanoTime() - tPrefill0;
                        long tSample0 = System.nanoTime();
                        next = tokenSampler().sample(logits, cfg, config, freq, rng);
                        if (profile != null)
                            profile.samplingNanos += System.nanoTime() - tSample0;
                    }

                    Set<Integer> stops = new HashSet<>();
                    for (int id : tokenizer.allStopTokenIds()) {
                        stops.add(id);
                    }
                    stops.addAll(config.eosTokenIds());
                    stops.addAll(cfg.stopTokenIds());

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        if (stops.contains(next) || emitter.isCancelled()) {
                            break;
                        }

                        markFirstToken(profile, requestStartNanos);
                        generatedTokenIds.add((long) next);
                        String delta = decoder.decodeNext((long) next);
                        if (delta == null) {
                            delta = "";
                        }

                        String fullGeneratedText = decoder.currentText();

                        if (!delta.isEmpty()) {
                            emitter.emit(InferenceResponse.builder()
                                    .requestId(requestId)
                                    .content(delta)
                                    .model(modelPath.getFileName().toString())
                                    .inputTokens(inputLen)
                                    .metadata("backend", "accelerate-safetensor")
                                    .metadata("prompt_token_source", "pretokenized")
                                    .metadata("conversation_kv_retained", true)
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
                            if (shouldStop) {
                                break;
                            }
                        }

                        recordSampleFrequency(freq, next);
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        int decodeStartPos = inputIds.length + step;
                        long tDecode0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().decodeLogitsTensor(next, decodeStartPos, model.weights(),
                                    config, arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().decode(next, decodeStartPos, model.weights(), config, arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = tokenSampler().sample(logits, cfg, config, freq, rng);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                    }

                    InferenceResponse.Builder builder = InferenceResponse.builder()
                            .requestId(requestId)
                            .content("")
                            .model(modelPath.getFileName().toString())
                            .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                            .finishReason(InferenceResponse.FinishReason.STOP)
                            .inputTokens(inputLen)
                            .metadata("backend", "accelerate-safetensor")
                            .metadata("prompt_token_source", "pretokenized")
                            .metadata("conversation_kv_retained", true);
                    if (profile != null) {
                        Map<String, Object> metadata = profile.metadata(backend, inputLen, generatedTokenIds.size());
                        metadata.forEach(builder::metadata);
                        maybePrintProfileSummary(profile, backend);
                    }
                    InferenceResponse finalResponse = builder.build();
                    if (onComplete != null) {
                        onComplete.accept(new DirectConversationTrace(
                                finalResponse,
                                inputIds.clone(),
                                toLongArray(generatedTokenIds),
                                session));
                        session = null;
                    }
                    emitter.emit(finalResponse);

                } catch (Throwable t) {
                    if (session != null) {
                        try {
                            session.close();
                        } catch (Exception ignored) {
                        }
                    }
                    log.error("Streaming conversation generation failed", t);
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

    public Multi<InferenceResponse> generateContinuationStreamWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete) {
        return generateContinuationStreamWithConversationTrace(
                fullInputIds,
                cachedPrefixTokens,
                session,
                modelPath,
                cfg,
                onComplete,
                null);
    }

    public Multi<InferenceResponse> generateContinuationStreamWithConversationTrace(
            long[] fullInputIds,
            int cachedPrefixTokens,
            KVCacheManager.KVCacheSession session,
            Path modelPath,
            GenerationConfig cfg,
            Consumer<DirectConversationTrace> onComplete,
            Integer replayTokenId) {
        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(STREAM_EXECUTOR_THREAD_FACTORY);
            executor.submit(() -> {
                Instant t0 = Instant.now();
                long requestStartNanos = System.nanoTime();
                String requestId = UUID.randomUUID().toString();
                int inputLen = fullInputIds.length;
                InferenceProfile profile = startProfile("stream");
                String backend = backendLabel(metalBackend);
                List<Long> generatedTokenIds = new ArrayList<>();

                try {
                    if (session == null) {
                        throw new IllegalArgumentException(
                                "Conversation continuation requires an active KV cache session");
                    }
                    if (cachedPrefixTokens < 0 || cachedPrefixTokens > fullInputIds.length) {
                        throw new IllegalArgumentException("Invalid cachedPrefixTokens: " + cachedPrefixTokens);
                    }

                    LoadedModel model = requireLoadedModel(modelPath);
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    ModelArchitecture arch = architectureRegistry().resolve(config);

                    long[] deltaInputIds = java.util.Arrays.copyOfRange(fullInputIds, cachedPrefixTokens,
                            fullInputIds.length);
                    boolean exactReplay = deltaInputIds.length == 0;
                    if (exactReplay && replayTokenId == null) {
                        throw new IllegalArgumentException(
                                "Conversation continuation requires at least one delta token or a replay token");
                    }
                    if (session.currentPos() != cachedPrefixTokens) {
                        throw new IllegalStateException("KV cache position " + session.currentPos()
                                + " does not match cached prefix tokens " + cachedPrefixTokens);
                    }

                    boolean directGreedy = canUseDirectGreedySampling(cfg);
                    int[] freq = directGreedy ? null : initializePromptFrequencies(config, fullInputIds);
                    Random rng = directGreedy ? null : new Random();

                    int next;
                    if (exactReplay) {
                        next = replayTokenId;
                    } else {
                        long tPrefill0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().prefillLogitsTensor(deltaInputIds, model.weights(),
                                    config, arch, session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().prefill(deltaInputIds, model.weights(), config, arch,
                                    session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            long tSample0 = System.nanoTime();
                            next = tokenSampler().sample(logits, cfg, config, freq, rng);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                    }

                    Set<Integer> stops = new HashSet<>();
                    for (int id : tokenizer.allStopTokenIds()) {
                        stops.add(id);
                    }
                    stops.addAll(config.eosTokenIds());
                    stops.addAll(cfg.stopTokenIds());

                    StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                    for (int step = 0; step < cfg.maxNewTokens(); step++) {
                        if (stops.contains(next) || emitter.isCancelled()) {
                            break;
                        }

                        markFirstToken(profile, requestStartNanos);
                        generatedTokenIds.add((long) next);
                        String delta = decoder.decodeNext((long) next);
                        if (delta == null) {
                            delta = "";
                        }

                        String fullGeneratedText = decoder.currentText();

                        if (!delta.isEmpty()) {
                            emitter.emit(InferenceResponse.builder()
                                    .requestId(requestId)
                                    .content(delta)
                                    .model(modelPath.getFileName().toString())
                                    .inputTokens(inputLen)
                                    .metadata("backend", "accelerate-safetensor")
                                    .metadata("prompt_token_source",
                                            exactReplay ? "conversation_replay" : "conversation_delta")
                                    .metadata("conversation_delta_prefill", !exactReplay)
                                    .metadata("conversation_exact_replay", exactReplay)
                                    .metadata("conversation_cached_prefix_tokens", cachedPrefixTokens)
                                    .metadata("conversation_delta_prompt_tokens", deltaInputIds.length)
                                    .metadata("conversation_kv_retained", true)
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
                            if (shouldStop) {
                                break;
                            }
                        }

                        recordSampleFrequency(freq, next);
                        if (step + 1 >= cfg.maxNewTokens()) {
                            break;
                        }
                        int decodeStartPos = fullInputIds.length + step;
                        long tDecode0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().decodeLogitsTensor(next, decodeStartPos, model.weights(),
                                    config, arch, session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().decode(next, decodeStartPos, model.weights(), config, arch,
                                    session);
                            if (profile != null) {
                                profile.decodeNanos += System.nanoTime() - tDecode0;
                                profile.decodeSteps++;
                            }
                            long tSample0 = System.nanoTime();
                            next = tokenSampler().sample(logits, cfg, config, freq, rng);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }
                    }

                    InferenceResponse.Builder builder = InferenceResponse.builder()
                            .requestId(requestId)
                            .content("")
                            .model(modelPath.getFileName().toString())
                            .durationMs(java.time.Duration.between(t0, Instant.now()).toMillis())
                            .finishReason(InferenceResponse.FinishReason.STOP)
                            .inputTokens(inputLen)
                            .metadata("backend", "accelerate-safetensor")
                            .metadata("prompt_token_source", exactReplay ? "conversation_replay" : "conversation_delta")
                            .metadata("conversation_delta_prefill", !exactReplay)
                            .metadata("conversation_exact_replay", exactReplay)
                            .metadata("conversation_cached_prefix_tokens", cachedPrefixTokens)
                            .metadata("conversation_delta_prompt_tokens", deltaInputIds.length)
                            .metadata("conversation_kv_retained", true);
                    if (profile != null) {
                        Map<String, Object> metadata = profile.metadata(backend, inputLen, generatedTokenIds.size());
                        metadata.forEach(builder::metadata);
                        maybePrintProfileSummary(profile, backend);
                    }
                    InferenceResponse finalResponse = builder.build();
                    if (onComplete != null) {
                        onComplete.accept(new DirectConversationTrace(
                                finalResponse,
                                fullInputIds.clone(),
                                toLongArray(generatedTokenIds),
                                session));
                    }
                    emitter.emit(finalResponse);

                } catch (Throwable t) {
                    log.error("Streaming conversation continuation failed", t);
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

    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(STREAM_EXECUTOR_THREAD_FACTORY);
            executor.submit(() -> {
                Instant t0 = Instant.now();
                long requestStartNanos = System.nanoTime();
                String requestId = UUID.randomUUID().toString();
                int inputLen = 0;
                InferenceProfile profile = startProfile("stream");
                String backend = backendLabel(metalBackend);

                try {
                    boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                    if (verbose) {
                        System.out.println("[DEBUG-S] 1: getLoadedModel");
                        System.out.flush();
                    }
                    LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
                    if (model == null) {
                        if (verbose) {
                            System.out.println("[DEBUG-S] 2: loadModel");
                            System.out.flush();
                        }
                        long tLoad0 = System.nanoTime();
                        loadModel(modelPath);
                        recordModelLoadNanos(System.nanoTime() - tLoad0);
                        model = (LoadedModel) getLoadedModel(modelPath);
                    }

                    if (model == null)
                        throw new RuntimeException("Model failed to load");

                    if (verbose) {
                        System.out.println("[DEBUG-S] 3: get tokenizer/config");
                        System.out.flush();
                    }
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    if (verbose) {
                        System.out.println("[DEBUG-S] 4: arch resolve");
                        System.out.flush();
                    }
                    ModelArchitecture arch = architectureRegistry().resolve(config);

                    if (verbose) {
                        System.out.println("[DEBUG-S] 5: tokenize");
                        System.out.flush();
                    }
                    long tTokenize0 = System.nanoTime();
                    long[] inputIds = tokenizer.encode(prompt, encodeOptionsFor(config, prompt));
                    if (profile != null)
                        profile.tokenizeNanos += System.nanoTime() - tTokenize0;
                    inputLen = inputIds.length;
                    if (verbose) {
                        System.out.printf("[DEBUG-S] 6: tokens=%d\n", inputLen);
                        System.out.flush();
                    }

                    Set<Integer> stops = new HashSet<>();
                    for (int id : tokenizer.allStopTokenIds())
                        stops.add(id);
                    stops.addAll(config.eosTokenIds());
                    stops.addAll(cfg.stopTokenIds());

                    int completionTokens = 0;
                    try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                            .createSession(cfg.maxKvCacheTokens())) {
                        if (verbose) {
                            System.out.println("[DEBUG-S] 7: allocate session");
                            System.out.flush();
                        }
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        if (profile != null)
                            profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                        if (verbose) {
                            System.out.println("[DEBUG-S] 8: prefill");
                            System.out.flush();
                        }
                        boolean directGreedy = canUseDirectGreedySampling(cfg);
                        int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                        Random rng = directGreedy ? null : new Random();
                        GreedySamplingMasks greedyMasks = directGreedy
                                ? buildGreedySamplingMasks(tokenizer, stops, config.vocabSize())
                                : null;

                        int next;
                        long tPrefill0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config,
                                    arch, session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            if (verbose) {
                                System.out.println("[DEBUG-S] 9: prefill done");
                                System.out.flush();
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config, tokenizer, true, stops, greedyMasks);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            if (verbose) {
                                System.out.println("[DEBUG-S] 9: prefill done");
                                System.out.flush();
                            }
                            long tSample0 = System.nanoTime();
                            next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, true, stops);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }

                        StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                        for (int step = 0; step < cfg.maxNewTokens(); step++) {

                            if (stops.contains(next) || emitter.isCancelled()) {
                                break;
                            }

                            markFirstToken(profile, requestStartNanos);
                            completionTokens++;

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

                            recordSampleFrequency(freq, next);
                            if (step + 1 >= cfg.maxNewTokens()) {
                                break;
                            }
                            int decodeStartPos = inputIds.length + step;
                            long tDecode0 = System.nanoTime();
                            if (directGreedy) {
                                AccelTensor logits = forwardPass().decodeLogitsTensor(next, decodeStartPos,
                                        model.weights(), config, arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                long tSample0 = System.nanoTime();
                                next = sampleGreedyFromTensor(logits, config, tokenizer, false, stops, greedyMasks);
                                if (profile != null)
                                    profile.samplingNanos += System.nanoTime() - tSample0;
                            } else {
                                float[] logits = forwardPass().decode(next, decodeStartPos, model.weights(), config,
                                        arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                long tSample0 = System.nanoTime();
                                next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, false, stops);
                                if (profile != null)
                                    profile.samplingNanos += System.nanoTime() - tSample0;
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
                        Map<String, Object> metadata = profile.metadata(backend, inputLen, completionTokens);
                        metadata.forEach(builder::metadata);
                        maybePrintProfileSummary(profile, backend);
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

    private static long[] toLongArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    public Multi<InferenceResponse> generateStream(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return Multi.createFrom().emitter(emitter -> {
            ExecutorService executor = Executors.newSingleThreadExecutor(STREAM_EXECUTOR_THREAD_FACTORY);
            executor.submit(() -> {
                Instant t0 = Instant.now();
                long requestStartNanos = System.nanoTime();
                String requestId = UUID.randomUUID().toString();
                int inputLen = 0;
                InferenceProfile profile = startProfile("stream");
                String backend = backendLabel(metalBackend);

                try {
                    LoadedModel model = requireLoadedModel(modelPath);
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    ModelArchitecture arch = architectureRegistry().resolve(config);

                    inputLen = inputIds.length;

                    int completionTokens = 0;
                    try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                            .createSession(cfg.maxKvCacheTokens())) {
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        if (profile != null)
                            profile.sessionAllocateNanos += System.nanoTime() - tAlloc0;

                        Set<Integer> stops = new HashSet<>();
                        for (int id : tokenizer.allStopTokenIds())
                            stops.add(id);
                        stops.addAll(config.eosTokenIds());
                        stops.addAll(cfg.stopTokenIds());

                        boolean directGreedy = canUseDirectGreedySampling(cfg);
                        int[] freq = directGreedy ? null : initializePromptFrequencies(config, inputIds);
                        Random rng = directGreedy ? null : new Random();
                        GreedySamplingMasks greedyMasks = directGreedy
                                ? buildGreedySamplingMasks(tokenizer, stops, config.vocabSize())
                                : null;

                        int next;
                        long tPrefill0 = System.nanoTime();
                        if (directGreedy) {
                            AccelTensor logits = forwardPass().prefillLogitsTensor(inputIds, model.weights(), config,
                                    arch, session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            long tSample0 = System.nanoTime();
                            next = sampleGreedyFromTensor(logits, config, tokenizer, true, stops, greedyMasks);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        } else {
                            float[] logits = forwardPass().prefill(inputIds, model.weights(), config, arch, session);
                            if (profile != null)
                                profile.prefillNanos += System.nanoTime() - tPrefill0;
                            long tSample0 = System.nanoTime();
                            next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, true, stops);
                            if (profile != null)
                                profile.samplingNanos += System.nanoTime() - tSample0;
                        }

                        StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                        for (int step = 0; step < cfg.maxNewTokens(); step++) {

                            if (stops.contains(next) || emitter.isCancelled()) {
                                break;
                            }

                            markFirstToken(profile, requestStartNanos);
                            completionTokens++;

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
                                        .metadata("prompt_token_source", "pretokenized")
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

                            recordSampleFrequency(freq, next);
                            if (step + 1 >= cfg.maxNewTokens()) {
                                break;
                            }
                            int decodeStartPos = inputIds.length + step;
                            long tDecode0 = System.nanoTime();
                            if (directGreedy) {
                                AccelTensor logits = forwardPass().decodeLogitsTensor(next, decodeStartPos,
                                        model.weights(), config, arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                long tSample0 = System.nanoTime();
                                next = sampleGreedyFromTensor(logits, config, tokenizer, false, stops, greedyMasks);
                                if (profile != null)
                                    profile.samplingNanos += System.nanoTime() - tSample0;
                            } else {
                                float[] logits = forwardPass().decode(next, decodeStartPos, model.weights(), config,
                                        arch,
                                        session);
                                if (profile != null) {
                                    profile.decodeNanos += System.nanoTime() - tDecode0;
                                    profile.decodeSteps++;
                                }
                                long tSample0 = System.nanoTime();
                                next = sampleNextToken(logits, tokenizer, cfg, config, freq, rng, false, stops);
                                if (profile != null)
                                    profile.samplingNanos += System.nanoTime() - tSample0;
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
                            .metadata("backend", "accelerate-safetensor")
                            .metadata("prompt_token_source", "pretokenized");
                    if (profile != null) {
                        Map<String, Object> metadata = profile.metadata(backend, inputLen, completionTokens);
                        metadata.forEach(builder::metadata);
                        maybePrintProfileSummary(profile, backend);
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
        if (quantizationEngine != null) {
            return quantizationEngine;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(QuantizationEngine.class);
                if (instance.isAvailable()) {
                    quantizationEngine = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (quantizationEngine == null) {
            throw new IllegalStateException("QuantizationEngine is not available");
        }
        return quantizationEngine;
    }

    public KVCacheManager getKVCacheManager() {
        if (kvCacheManager != null) {
            return kvCacheManager;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(KVCacheManager.class);
                if (instance.isAvailable()) {
                    kvCacheManager = instance.get();
                }
            }
        } catch (Exception ignored) {
        }
        if (kvCacheManager == null) {
            throw new IllegalStateException("KVCacheManager is not available");
        }
        return kvCacheManager;
    }

    private LoadedModel requireLoadedModel(Path modelPath) {
        LoadedModel model = (LoadedModel) getLoadedModel(modelPath);
        if (model == null) {
            long tLoad0 = System.nanoTime();
            loadModel(modelPath);
            recordModelLoadNanos(System.nanoTime() - tLoad0);
            model = (LoadedModel) getLoadedModel(modelPath);
        }
        if (model == null) {
            throw new RuntimeException("Model failed to load");
        }
        return model;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weight loading — pure AccelTensor, no LibTorch
    // ─────────────────────────────────────────────────────────────────────────

    private WeightLoadResult loadWeights(Path modelPath, QuantizationEngine.QuantStrategy quantStrategy) {
        log.debugf("DirectInferenceEngine: opening weights from [%s]", modelPath);

        Map<String, AccelTensor> weights;
        try (SafetensorShardSession session = safetensorLoader().open(modelPath)) {
            weights = weightBridge().bridgeAll(session);
        } catch (Exception e) {
            log.errorf("Failed to load weights from %s: %s", modelPath, e.getMessage());
            throw new IllegalStateException("Failed to load weights: " + e.getMessage(), e);
        }

        String quantCacheState = "off";
        Path quantCachePath = null;
        if (quantStrategy == null || quantStrategy == QuantizationEngine.QuantStrategy.NONE) {
            weights = maybeUseMetalF16DiskCache(modelPath, weights);
        }
        if (quantStrategy != null && quantStrategy != QuantizationEngine.QuantStrategy.NONE) {
            QuantizationEngine.InferenceQuantizationResult quantizationResult = getQuantizationEngine()
                    .quantizeWeightsForInferenceDetailed(modelPath, weights, quantStrategy);
            weights = quantizationResult.weights();
            quantCacheState = quantizationResult.cacheState();
            quantCachePath = quantizationResult.cachePath();
        }

        applyWeightAliases(weights);
        log.infof("DirectInferenceEngine: bridged %d tensors (Accelerate/FFM backend, strategy=%s)",
                weights.size(), quantStrategy);
        return new WeightLoadResult(weights, quantCacheState, quantCachePath);
    }

    private Map<String, AccelTensor> maybeUseMetalF16DiskCache(Path modelPath, Map<String, AccelTensor> weights) {
        if (Boolean.getBoolean(DISABLE_METAL_F16_DISK_CACHE_PROPERTY) || !isNativeMetalRuntimeActive()) {
            return weights;
        }
        long cacheMaxBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MAX_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MAX_BYTES);
        if (cacheMaxBytes <= 0L) {
            return weights;
        }
        Path cachePath = metalF16CachePath(modelPath);
        long cacheBudgetBytes = Files.isRegularFile(cachePath)
                ? cacheMaxBytes
                : availableMetalF16CacheBudget(cachePath, cacheMaxBytes);
        Map<String, AccelTensor> cacheableWeights = selectMetalF16CacheWeights(weights, cacheBudgetBytes);
        if (cacheableWeights.isEmpty()) {
            return weights;
        }
        long cacheBytes = estimateMetalF16CacheBytes(cacheableWeights);
        if (cacheBytes <= 0L || cacheBytes > cacheMaxBytes) {
            return weights;
        }

        Map<String, AccelTensor> cached = tryLoadMetalF16DiskCache(cachePath, weights, cacheableWeights.keySet());
        if (cached != null) {
            return cached;
        }
        if (!hasEnoughMetalF16CacheSpace(cachePath, cacheBytes)) {
            return weights;
        }

        Path tempPath = metalF16TempCachePath(cachePath);
        try {
            Files.createDirectories(cachePath.getParent());
            Map<String, AccelTensor> converted = new LinkedHashMap<>(cacheableWeights.size() * 2);
            try {
                for (Map.Entry<String, AccelTensor> entry : cacheableWeights.entrySet()) {
                    AccelTensor tensor = entry.getValue();
                    AccelTensor cacheTensor = tensor.quantType() == AccelTensor.QuantType.BF16
                            ? tensor.toF16CachedUpTo(cacheMaxBytes)
                            : tensor;
                    if (cacheTensor == null) {
                        return weights;
                    }
                    converted.put(entry.getKey(), cacheTensor);
                }

                Files.deleteIfExists(tempPath);
                SafetensorWriter.save(tempPath, converted);
                moveReplacing(tempPath, cachePath);
                log.infof("DirectInferenceEngine: saved Metal F16 weight cache [%s]", cachePath);
                cached = tryLoadMetalF16DiskCache(cachePath, weights, cacheableWeights.keySet());
                if (cached != null) {
                    return cached;
                }
            } finally {
            // Converted tensors are cached on the original BF16 tensors, so the normal model unload path owns them.
            }
        } catch (Exception e) {
            try {
                Files.deleteIfExists(tempPath);
            } catch (IOException cleanupError) {
                e.addSuppressed(cleanupError);
            }
            log.warnf(e, "Failed to persist Metal F16 weight cache for %s", modelPath);
        }
        return weights;
    }

    private Map<String, AccelTensor> tryLoadMetalF16DiskCache(
            Path cachePath,
            Map<String, AccelTensor> expected,
            Set<String> expectedCachedNames) {
        if (!Files.isRegularFile(cachePath)) {
            return null;
        }
        try (SafetensorShardSession cacheSession = safetensorLoader().open(cachePath)) {
            Map<String, AccelTensor> cached = weightBridge().bridgeAll(cacheSession);
            if (!cached.keySet().containsAll(expectedCachedNames)) {
                cached.values().forEach(AccelTensor::close);
                log.warnf("Ignoring stale Metal F16 weight cache with missing tensors: %s", cachePath);
                return null;
            }

            Map<String, AccelTensor> merged = new LinkedHashMap<>(expected);
            Set<String> usedCachedNames = new HashSet<>();
            for (String name : expectedCachedNames) {
                AccelTensor source = expected.get(name);
                AccelTensor replacement = cached.get(name);
                if (!isValidMetalF16CacheReplacement(source, replacement)) {
                    cached.values().forEach(AccelTensor::close);
                    log.warnf("Ignoring stale Metal F16 weight cache with mismatched tensor '%s': %s", name, cachePath);
                    return null;
                }
                AccelTensor old = merged.put(name, replacement);
                if (old != null) {
                    old.close();
                }
                usedCachedNames.add(name);
            }

            cached.entrySet().stream()
                    .filter(entry -> !usedCachedNames.contains(entry.getKey()))
                    .map(Map.Entry::getValue)
                    .forEach(AccelTensor::close);
            log.infof("DirectInferenceEngine: using Metal F16 weight cache [%s] (%d tensors)",
                    cachePath, usedCachedNames.size());
            return merged;
        } catch (Exception e) {
            log.warnf(e, "Failed to load Metal F16 weight cache from %s", cachePath);
        }
        return null;
    }

    private boolean isValidMetalF16CacheReplacement(AccelTensor source, AccelTensor cached) {
        if (source == null || cached == null || !Arrays.equals(source.shape(), cached.shape())) {
            return false;
        }
        if (source.quantType() != AccelTensor.QuantType.BF16
                && source.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        return cached.quantType() == AccelTensor.QuantType.F16;
    }

    private Map<String, AccelTensor> selectMetalF16CacheWeights(Map<String, AccelTensor> weights, long cacheBudgetBytes) {
        Map<String, AccelTensor> selected = new LinkedHashMap<>();
        long usedBytes = 0L;
        List<Map.Entry<String, AccelTensor>> candidates = weights.entrySet().stream()
                .filter(entry -> isMetalF16DiskCacheCandidate(entry.getKey(), entry.getValue()))
                .sorted(Comparator.comparingInt(entry -> metalF16CachePriority(entry.getKey())))
                .toList();
        for (Map.Entry<String, AccelTensor> entry : candidates) {
            String name = entry.getKey();
            AccelTensor tensor = entry.getValue();
            long bytes = metalF16CacheByteSize(tensor);
            if (cacheBudgetBytes > 0L && usedBytes + bytes > cacheBudgetBytes) {
                continue;
            }
            selected.put(name, tensor);
            usedBytes += bytes;
        }
        return selected;
    }

    private boolean isMetalF16DiskCacheCandidate(String name, AccelTensor tensor) {
        if (tensor == null || tensor.rank() < 2) {
            return false;
        }
        if (tensor.quantType() != AccelTensor.QuantType.BF16
                && tensor.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        if (name.startsWith("model.audio_tower.")
                || name.startsWith("model.vision_tower.")
                || name.startsWith("model.embed_audio.")
                || name.startsWith("model.embed_vision.")) {
            return false;
        }
        return !name.endsWith("embed_tokens_per_layer.weight");
    }

    private int metalF16CachePriority(String name) {
        if (name.endsWith("embed_tokens.weight")) {
            return 0;
        }
        if (name.contains(".mlp.") || name.contains(".feed_forward.")) {
            return 1;
        }
        if (name.contains(".self_attn.") || name.contains(".attention.")) {
            return 2;
        }
        if (name.startsWith("model.language_model.layers.") || name.startsWith("model.layers.")) {
            return 3;
        }
        return 4;
    }

    private long availableMetalF16CacheBudget(Path cachePath, long cacheMaxBytes) {
        Path cacheDir = cachePath.getParent();
        if (cacheDir == null) {
            return cacheMaxBytes;
        }
        long minFreeBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES);
        if (minFreeBytes < 0L) {
            minFreeBytes = 0L;
        }
        try {
            Files.createDirectories(cacheDir);
            long usableBytes = Files.getFileStore(cacheDir).getUsableSpace();
            long reclaimableBytes = staleMetalF16TempCacheBytes(cachePath);
            long effectiveUsableBytes = saturatingAdd(usableBytes, reclaimableBytes);
            long reserveBytes = saturatingAdd(minFreeBytes, METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES);
            long diskBudgetBytes = Math.max(0L, effectiveUsableBytes - reserveBytes);
            return Math.min(cacheMaxBytes, diskBudgetBytes);
        } catch (IOException e) {
            log.warnf(e, "Could not determine Metal F16 cache budget for %s", cacheDir);
            return cacheMaxBytes;
        }
    }

    private boolean hasEnoughMetalF16CacheSpace(Path cachePath, long cacheBytes) {
        Path cacheDir = cachePath.getParent();
        if (cacheDir == null) {
            return true;
        }
        long minFreeBytes = Long.getLong(
                METAL_F16_DISK_CACHE_MIN_FREE_BYTES_PROPERTY,
                DEFAULT_METAL_F16_DISK_CACHE_MIN_FREE_BYTES);
        if (minFreeBytes < 0L) {
            minFreeBytes = 0L;
        }
        try {
            Files.createDirectories(cacheDir);
            long usableBytes = Files.getFileStore(cacheDir).getUsableSpace();
            long reclaimableBytes = staleMetalF16TempCacheBytes(cachePath);
            long effectiveUsableBytes = saturatingAdd(usableBytes, reclaimableBytes);
            long reserveBytes = saturatingAdd(minFreeBytes, METAL_F16_DISK_CACHE_WRITE_SAFETY_BYTES);
            long requiredBytes = Math.addExact(cacheBytes, reserveBytes);
            if (effectiveUsableBytes >= requiredBytes) {
                return true;
            }
            log.warnf(
                    "Skipping Metal F16 weight cache because free space is too low. Need ~%s for cache plus %s reserve, have %s%s.",
                    formatGib(cacheBytes),
                    formatGib(reserveBytes),
                    formatGib(usableBytes),
                    reclaimableBytes > 0L ? " (+" + formatGib(reclaimableBytes) + " reclaimable temp)" : "");
        } catch (ArithmeticException e) {
            log.warnf("Skipping Metal F16 weight cache because required space overflowed for %s", cachePath);
        } catch (IOException e) {
            log.warnf(e, "Skipping Metal F16 weight cache because cache directory space could not be checked: %s", cacheDir);
        }
        return false;
    }

    private Path metalF16TempCachePath(Path cachePath) {
        return cachePath.resolveSibling(cachePath.getFileName() + ".tmp");
    }

    private long staleMetalF16TempCacheBytes(Path cachePath) {
        Path tempPath = metalF16TempCachePath(cachePath);
        try {
            return Files.isRegularFile(tempPath) ? Files.size(tempPath) : 0L;
        } catch (IOException e) {
            return 0L;
        }
    }

    private long saturatingAdd(long left, long right) {
        try {
            return Math.addExact(left, right);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }

    private String formatGib(long bytes) {
        return String.format(Locale.ROOT, "%.1f GiB", bytes / (double) GIB);
    }

    private void moveReplacing(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private boolean allWeightsAreMetalF16Cacheable(Map<String, AccelTensor> weights) {
        boolean sawHalfMatrix = false;
        for (AccelTensor tensor : weights.values()) {
            if (tensor == null) {
                return false;
            }
            AccelTensor.QuantType type = tensor.quantType();
            if (type == AccelTensor.QuantType.BF16 || type == AccelTensor.QuantType.F16) {
                if (tensor.rank() >= 2) {
                    sawHalfMatrix = true;
                }
                continue;
            }
            if (type != AccelTensor.QuantType.F32) {
                return false;
            }
        }
        return sawHalfMatrix;
    }

    private long estimateMetalF16CacheBytes(Map<String, AccelTensor> weights) {
        long total = 0L;
        for (AccelTensor tensor : weights.values()) {
            if (tensor == null) {
                return Long.MAX_VALUE;
            }
            long bytes = metalF16CacheByteSize(tensor);
            try {
                total = Math.addExact(total, bytes);
            } catch (ArithmeticException e) {
                return Long.MAX_VALUE;
            }
        }
        return total;
    }

    private long metalF16CacheByteSize(AccelTensor tensor) {
        return switch (tensor.quantType()) {
            case BF16, F16 -> tensor.halfStorageByteSize();
            case F32 -> tensor.dequantizedByteSize();
            case INT8, FP8 -> tensor.numel();
            case INT4 -> (tensor.numel() + 1L) / 2L;
        };
    }

    private boolean sameWeightLayout(Map<String, AccelTensor> expected, Map<String, AccelTensor> actual) {
        if (!expected.keySet().equals(actual.keySet())) {
            return false;
        }
        for (Map.Entry<String, AccelTensor> entry : expected.entrySet()) {
            AccelTensor source = entry.getValue();
            AccelTensor cached = actual.get(entry.getKey());
            if (cached == null || !Arrays.equals(source.shape(), cached.shape())) {
                return false;
            }
            AccelTensor.QuantType sourceType = source.quantType();
            AccelTensor.QuantType cachedType = cached.quantType();
            if (sourceType == AccelTensor.QuantType.BF16 && source.rank() >= 2) {
                if (cachedType != AccelTensor.QuantType.F16) {
                    return false;
                }
            } else if (sourceType != cachedType) {
                return false;
            }
        }
        return true;
    }

    private Path metalF16CachePath(Path modelPath) {
        String gollekHome = System.getProperty("gollek.home",
                Path.of(System.getProperty("user.home"), ".gollek").toString());
        String key = sha256Hex(modelCacheFingerprint(modelPath));
        return Path.of(gollekHome, "cache", "metal-f16", key + ".safetensors");
    }

    private String modelCacheFingerprint(Path modelPath) {
        Path resolved = modelPath.toAbsolutePath().normalize();
        StringBuilder fingerprint = new StringBuilder(resolved.toString());
        try {
            if (Files.isRegularFile(resolved)) {
                appendFileFingerprint(fingerprint, resolved);
            } else if (Files.isDirectory(resolved)) {
                try (var stream = Files.walk(resolved, 2)) {
                    List<Path> safetensors = stream
                            .filter(Files::isRegularFile)
                            .filter(this::isSafetensorFile)
                            .sorted()
                            .toList();
                    for (Path path : safetensors) {
                        appendFileFingerprint(fingerprint, path);
                    }
                }
            }
        } catch (IOException e) {
            log.warnf(e, "Falling back to path-only Metal F16 cache key for %s", resolved);
        }
        return fingerprint.toString();
    }

    private boolean isSafetensorFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private void appendFileFingerprint(StringBuilder fingerprint, Path path) throws IOException {
        fingerprint.append('|')
                .append(path.toAbsolutePath().normalize())
                .append(':')
                .append(Files.size(path))
                .append(':')
                .append(Files.getLastModifiedTime(path).toMillis());
    }

    private String sha256Hex(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(Character.forDigit((b >>> 4) & 0xF, 16));
                out.append(Character.forDigit(b & 0xF, 16));
            }
            return out.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
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
                return ModelConfig.fromDirectory(configDir, resolvedObjectMapper());
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

    private record GreedySamplingMasks(java.util.BitSet firstStep, java.util.BitSet continuation) {
        java.util.BitSet maskFor(boolean isFirstStep) {
            java.util.BitSet mask = isFirstStep ? firstStep : continuation;
            return mask == null ? new java.util.BitSet() : mask;
        }
    }

    private static GreedySamplingMasks buildGreedySamplingMasks(Tokenizer tokenizer, Set<Integer> stops,
            int vocabSize) {
        return new GreedySamplingMasks(
                buildDisallowedTokenMask(tokenizer, true, stops, vocabSize),
                buildDisallowedTokenMask(tokenizer, false, stops, vocabSize));
    }

    private static int[] initializePromptFrequencies(ModelConfig config, long[] tokenIds) {
        int[] freq = new int[config.vocabSize()];
        if (tokenIds == null || tokenIds.length == 0) {
            return freq;
        }
        for (long tokenId : tokenIds) {
            if (tokenId >= 0 && tokenId < freq.length) {
                freq[(int) tokenId]++;
            }
        }
        return freq;
    }

    private static void recordSampleFrequency(int[] freq, int tokenId) {
        if (freq != null && tokenId >= 0 && tokenId < freq.length) {
            freq[tokenId]++;
        }
    }

    private static boolean isGemma4Text(ModelConfig config) {
        if (config == null || config.modelType() == null) {
            return false;
        }
        String modelType = config.modelType().toLowerCase();
        return modelType.startsWith("gemma4");
    }

    private int sampleNextToken(float[] logits, Tokenizer tokenizer, GenerationConfig cfg, ModelConfig config,
            int[] freq, Random rng, boolean firstStep, Set<Integer> stops) {
        maskDisallowedContinuationTokens(logits, tokenizer, firstStep, stops);
        for (int attempt = 0; attempt < 8; attempt++) {
            int next = tokenSampler().sample(logits, cfg, config, freq, rng);
            if (!shouldRejectSampledToken(next, tokenizer, config, firstStep, stops)) {
                debugSampleChoice("accept", next, tokenizer, firstStep, attempt);
                return next;
            }
            debugSampleChoice("reject", next, tokenizer, firstStep, attempt);
            if (next >= 0 && next < logits.length) {
                logits[next] = Float.NEGATIVE_INFINITY;
            }
        }
        int fallback = tokenSampler().sample(logits, cfg, config, freq, rng);
        debugSampleChoice("fallback", fallback, tokenizer, firstStep, 8);
        return fallback;
    }

    private static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config) {
        return sampleGreedyFromTensor(logits, config, null, false, java.util.Collections.emptySet());
    }

    private static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            boolean firstStep, Set<Integer> stops) {
        return sampleGreedyFromTensor(logits, config, tokenizer, firstStep, stops, null);
    }

    private static int sampleGreedyFromTensor(AccelTensor logits, ModelConfig config, Tokenizer tokenizer,
            boolean firstStep, Set<Integer> stops, GreedySamplingMasks masks) {
        try {
            java.lang.foreign.MemorySegment seg = logits.dataPtr();
            long vocab = logits.numel();
            Double cap = config.finalLogitSoftcapping();
            float softCap = cap != null && cap > 0 ? cap.floatValue() : 0.0f;
            java.util.BitSet baseMask = masks == null
                    ? buildDisallowedTokenMask(tokenizer, firstStep, stops, (int) vocab)
                    : masks.maskFor(firstStep);
            int[] rejectedCandidates = null;
            int rejectedCount = 0;

            for (int attempt = 0; attempt < 8; attempt++) {
                if (attempt == 0) {
                    debugTopGreedyCandidates(seg, vocab, softCap, tokenizer, config, firstStep, stops, 8);
                }
                int best = -1;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < vocab; i++) {
                    if (rejectedCount > 0 && containsRejectedCandidate(rejectedCandidates, rejectedCount, i)) {
                        continue;
                    }
                    // Logit softcap is monotonic, so it cannot change greedy argmax ordering.
                    float value = seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
                    if (value > bestVal) {
                        bestVal = value;
                        best = i;
                    }
                }
                if (best < 0) {
                    break;
                }
                if (!isGreedyCandidateRejected(best, baseMask, tokenizer, config, firstStep, stops)) {
                    debugSampleChoice("accept", best, tokenizer, firstStep, attempt);
                    return best;
                }
                debugSampleChoice("reject", best, tokenizer, firstStep, attempt);
                if (rejectedCandidates == null) {
                    rejectedCandidates = new int[8];
                }
                rejectedCandidates[rejectedCount++] = best;
            }
            return -1;
        } finally {
            logits.close();
        }
    }

    private static boolean isGreedyCandidateRejected(int tokenId,
            java.util.BitSet baseMask,
            Tokenizer tokenizer,
            ModelConfig config,
            boolean firstStep,
            Set<Integer> stops) {
        if (tokenId < 0) {
            return false;
        }
        if (baseMask != null && baseMask.get(tokenId)) {
            return true;
        }
        return shouldRejectSampledToken(tokenId, tokenizer, config, firstStep, stops);
    }

    private static boolean containsRejectedCandidate(int[] rejectedCandidates, int rejectedCount, int tokenId) {
        if (rejectedCandidates == null) {
            return false;
        }
        for (int i = 0; i < rejectedCount; i++) {
            if (rejectedCandidates[i] == tokenId) {
                return true;
            }
        }
        return false;
    }

    private static void maskDisallowedContinuationTokens(float[] logits, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops) {
        if (logits == null || tokenizer == null) {
            return;
        }
        java.util.BitSet mask = buildDisallowedTokenMask(tokenizer, firstStep, stops, logits.length);
        for (int tokenId = mask.nextSetBit(0); tokenId >= 0; tokenId = mask.nextSetBit(tokenId + 1)) {
            logits[tokenId] = Float.NEGATIVE_INFINITY;
        }
    }

    private static java.util.BitSet buildDisallowedTokenMask(Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops, int vocabSize) {
        java.util.BitSet mask = new java.util.BitSet(Math.max(0, vocabSize));
        if (tokenizer == null || vocabSize <= 0) {
            return mask;
        }
        setIfInVocab(mask, tokenizer.bosTokenId(), vocabSize);
        setIfInVocab(mask, tokenizer.padTokenId(), vocabSize);

        Map<String, Integer> specialTokens = tokenizer.specialTokens();
        if (specialTokens != null && !specialTokens.isEmpty()) {
            for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
                Integer id = entry.getValue();
                if (id == null || id < 0 || id >= vocabSize) {
                    continue;
                }
                String text = entry.getKey();
                if (text != null && isAllowedGemma4ControlText(text.trim())) {
                    continue;
                }
                if (stops == null || !stops.contains(id)) {
                    mask.set(id);
                }
            }
        }

        if (firstStep && stops != null) {
            for (Integer stop : stops) {
                if (stop != null) {
                    setIfInVocab(mask, stop, vocabSize);
                }
            }
        }
        return mask;
    }

    private static void setIfInVocab(java.util.BitSet mask, int tokenId, int vocabSize) {
        if (tokenId >= 0 && tokenId < vocabSize) {
            mask.set(tokenId);
        }
    }

    private static boolean isDisallowedContinuationToken(int tokenId, Tokenizer tokenizer, boolean firstStep,
            Set<Integer> stops) {
        if (tokenizer == null || tokenId < 0) {
            return false;
        }
        if (tokenId == tokenizer.bosTokenId() || tokenId == tokenizer.padTokenId()) {
            return true;
        }
        String specialText = specialTokenText(tokenizer, tokenId);
        if (specialText != null) {
            if (isAllowedGemma4ControlText(specialText.trim())) {
                return false;
            }
            return stops == null || !stops.contains(tokenId);
        }
        return firstStep && stops != null && stops.contains(tokenId);
    }

    private static boolean shouldRejectSampledToken(int tokenId, Tokenizer tokenizer, ModelConfig config,
            boolean firstStep, Set<Integer> stops) {
        if (tokenId < 0) {
            return false;
        }
        if (isDisallowedContinuationToken(tokenId, tokenizer, firstStep, stops)) {
            return true;
        }
        boolean gemma4 = isGemma4Text(config);
        if (!firstStep && !gemma4) {
            return false;
        }
        if (tokenizer == null) {
            return false;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        if (decoded == null) {
            return false;
        }
        String trimmed = decoded.trim();
        if (isAllowedGemma4ControlText(trimmed)) {
            return false;
        }
        if (firstStep && gemma4
                && ("model".equalsIgnoreCase(trimmed) || "assistant".equalsIgnoreCase(trimmed))) {
            return true;
        }
        // Reject only truly empty decodes, not whitespace (space/newline) — those are
        // valid first tokens.
        if (gemma4 && decoded.isEmpty()) {
            return true;
        }
        if (!firstStep) {
            return false;
        }
        return trimmed.startsWith("<|")
                || trimmed.endsWith("|>")
                || (trimmed.startsWith("<") && trimmed.endsWith(">"))
                || trimmed.startsWith("<unused");
    }

    private static boolean isAllowedGemma4ControlToken(Tokenizer tokenizer, int tokenId) {
        String specialText = specialTokenText(tokenizer, tokenId);
        return specialText != null && isAllowedGemma4ControlText(specialText.trim());
    }

    private static String specialTokenText(Tokenizer tokenizer, int tokenId) {
        if (tokenizer == null || tokenId < 0) {
            return null;
        }
        Map<String, Integer> specialTokens = tokenizer.specialTokens();
        if (specialTokens == null || specialTokens.isEmpty()) {
            return null;
        }
        for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
            Integer id = entry.getValue();
            if (id != null && id == tokenId) {
                return entry.getKey();
            }
        }
        return null;
    }

    private static boolean isAllowedGemma4ControlText(String text) {
        return "<|channel>".equals(text)
                || "<channel|>".equals(text)
                || "<|think|>".equals(text);
    }

    private static String decodeSingleToken(Tokenizer tokenizer, int tokenId, boolean skipSpecialTokens) {
        try {
            DecodeOptions options = DecodeOptions.defaultOptions().skipSpecialTokens(skipSpecialTokens);
            return tokenizer.decode(new long[] { tokenId }, options);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void debugSampleChoice(String decision, int tokenId, Tokenizer tokenizer, boolean firstStep,
            int attempt) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
        System.out.printf("[DEBUG-SAMPLE] %s firstStep=%s attempt=%d token=%d text=%s%n",
                decision, firstStep, attempt, tokenId, printable);
        System.out.flush();
    }

    private static void debugTopGreedyCandidates(java.lang.foreign.MemorySegment seg,
            long vocab,
            float softCap,
            Tokenizer tokenizer,
            ModelConfig config,
            boolean firstStep,
            Set<Integer> stops,
            int limit) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        int topN = Math.max(1, limit);
        int[] ids = new int[topN];
        float[] vals = new float[topN];
        java.util.Arrays.fill(ids, -1);
        java.util.Arrays.fill(vals, Float.NEGATIVE_INFINITY);

        for (int i = 0; i < vocab; i++) {
            float value = applyLogitSoftcap(seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i), softCap);
            for (int slot = 0; slot < topN; slot++) {
                if (value > vals[slot]) {
                    for (int shift = topN - 1; shift > slot; shift--) {
                        vals[shift] = vals[shift - 1];
                        ids[shift] = ids[shift - 1];
                    }
                    vals[slot] = value;
                    ids[slot] = i;
                    break;
                }
            }
        }

        System.out.printf("[DEBUG-TOP] firstStep=%s%n", firstStep);
        for (int rank = 0; rank < topN; rank++) {
            int tokenId = ids[rank];
            if (tokenId < 0) {
                continue;
            }
            boolean disallowed = isDisallowedContinuationToken(tokenId, tokenizer, firstStep, stops);
            boolean rejected = shouldRejectSampledToken(tokenId, tokenizer, config, firstStep, stops);
            String decoded = decodeSingleToken(tokenizer, tokenId, false);
            String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
            System.out.printf("[DEBUG-TOP] rank=%d token=%d logit=%f disallowed=%s rejected=%s text=%s%n",
                    rank + 1, tokenId, vals[rank], disallowed, rejected, printable);
        }
        System.out.flush();
    }

    private static void debugTokenSequence(Tokenizer tokenizer, long[] tokenIds, String label) {
        System.out.printf("[DEBUG-TOKENS] %s:", label);
        for (long tokenId : tokenIds) {
            String decoded = decodeSingleToken(tokenizer, (int) tokenId, false);
            String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
            System.out.printf(" [%d:%s]", tokenId, printable);
        }
        System.out.println();
        System.out.flush();
    }

    private static void debugChosenToken(Tokenizer tokenizer, int tokenId, int step) {
        if (!Boolean.getBoolean("gollek.verbose")) {
            return;
        }
        String decoded = decodeSingleToken(tokenizer, tokenId, false);
        String printable = decoded == null ? "<null>" : decoded.replace("\n", "\\n");
        System.out.printf("[DEBUG-CHOSEN] step=%d token=%d text=%s%n", step, tokenId, printable);
        System.out.flush();
    }

    private static float applyLogitSoftcap(float value, float softCap) {
        if (softCap <= 0.0f) {
            return value;
        }
        return (float) (softCap * Math.tanh(value / softCap));
    }

    private static EncodeOptions encodeOptionsFor(ModelConfig config) {
        return encodeOptionsFor(config, null);
    }

    private static EncodeOptions encodeOptionsFor(ModelConfig config, String prompt) {
        EncodeOptions options = EncodeOptions.defaultOptions();
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        if (modelType.startsWith("gemma4")) {
            options.addBos = false;
        } else if (modelType.startsWith("gemma")) {
            options.addBos = !looksLikeGemmaTurnPrompt(prompt);
        }
        return options;
    }

    private static boolean looksLikeGemmaTurnPrompt(String prompt) {
        if (prompt == null) {
            return false;
        }
        String trimmed = prompt.stripLeading();
        if (trimmed.startsWith("<bos>")) {
            trimmed = trimmed.substring("<bos>".length()).stripLeading();
        }
        return trimmed.startsWith("<start_of_turn>")
                || trimmed.startsWith("<|turn>");
    }
}
