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

import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.debugChosenToken;
import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.backendLabel;
import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.clearProfile;
import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.runWithProfileSuspended;
import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.startProfile;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.quantization.bridge.AccelWeightBridge;
import tech.kayys.gollek.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.model.registry.ModelArchitectureRegistry;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.warmup.ModelWarmupService;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Instant;
import java.util.function.Supplier;
import java.util.function.Consumer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.Map;
import java.util.Collection;
import java.util.concurrent.ThreadFactory;

@ApplicationScoped
public class DirectInferenceEngine implements SafetensorEngine {
    private static final ThreadFactory STREAM_EXECUTOR_THREAD_FACTORY = runnable -> {
        Thread thread = new Thread(runnable, "gollek-direct-stream");
        thread.setDaemon(true);
        return thread;
    };
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
    private DirectGenerationStepSampler stepSampler;
    private DirectGenerationLoop generationLoop;
    private DirectGenerationExecutor generationExecutor;

    @Inject
    ModelArchitectureRegistry archRegistry;

    @Inject
    KVCacheManager kvCacheManager;

    @Inject
    ModelWarmupService modelWarmupService;

    private final DirectLoadedModelRegistry<LoadedModel> loadedModels = new DirectLoadedModelRegistry<>();
    private DirectModelLoader modelLoader;

    private ModelArchitectureRegistry architectureRegistry() {
        archRegistry = DirectInferenceDependencies.resolve(
                archRegistry, ModelArchitectureRegistry.class, "ModelArchitectureRegistry");
        return archRegistry;
    }

    private DirectForwardPass forwardPass() {
        forwardPass = DirectInferenceDependencies.resolve(
                forwardPass, DirectForwardPass.class, "DirectForwardPass");
        return forwardPass;
    }

    private TokenSampler tokenSampler() {
        tokenSampler = DirectInferenceDependencies.resolve(tokenSampler, TokenSampler.class, "TokenSampler");
        return tokenSampler;
    }

    private DirectGenerationStepSampler stepSampler() {
        if (stepSampler == null) {
            stepSampler = new DirectGenerationStepSampler(this::forwardPass, this::tokenSampler);
        }
        return stepSampler;
    }

    private DirectGenerationLoop generationLoop() {
        if (generationLoop == null) {
            generationLoop = new DirectGenerationLoop(this::stepSampler);
        }
        return generationLoop;
    }

    private DirectGenerationExecutor generationExecutor() {
        if (generationExecutor == null) {
            generationExecutor = new DirectGenerationExecutor(this::stepSampler, this::generationLoop);
        }
        return generationExecutor;
    }

    private SafetensorLoaderFacade safetensorLoader() {
        safetensorLoader = DirectInferenceDependencies.resolve(
                safetensorLoader, SafetensorLoaderFacade.class, "SafetensorLoaderFacade");
        return safetensorLoader;
    }

    private AccelWeightBridge weightBridge() {
        bridge = DirectInferenceDependencies.resolve(bridge, AccelWeightBridge.class, "AccelWeightBridge");
        return bridge;
    }

    private ObjectMapper resolvedObjectMapper() {
        objectMapper = DirectInferenceDependencies.resolve(objectMapper, ObjectMapper.class, "ObjectMapper");
        return objectMapper;
    }

    private DirectModelLoader modelLoader() {
        if (modelLoader == null) {
            modelLoader = new DirectModelLoader(
                    this::safetensorLoader,
                    this::weightBridge,
                    this::resolvedObjectMapper,
                    this::getQuantizationEngine,
                    this::architectureRegistry);
        }
        return modelLoader;
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

    /** Compatibility type; loaded-model state lives in {@link DirectLoadedModel}. */
    public static class LoadedModel extends DirectLoadedModel {
        public LoadedModel(Path path, Map<String, AccelTensor> weights,
                Tokenizer tokenizer, String key,
                boolean quantized, QuantizationEngine.QuantStrategy quantStrategy,
                String quantCacheState, Path quantCachePath,
                ModelConfig config, ModelArchitecture architecture, ModelRuntimeTraits runtimeTraits,
                Arena weightArena) {
            super(path, weights, tokenizer, key, quantized, quantStrategy, quantCacheState, quantCachePath,
                    config, architecture, runtimeTraits, weightArena);
        }
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

        QuantizationEngine.QuantStrategy effectiveQuantStrategy =
                DirectModelLoader.normalizeQuantStrategy(quantStrategy);
        Path resolved = loadedModels.resolve(modelPath);

        LoadedModel existing = loadedModels.find(resolved);
        if (existing != null && existing.getQuantStrategy() == effectiveQuantStrategy) {
            log.infof("DirectInferenceEngine: model already loaded [%s] (strategy=%s)",
                    resolved.getFileName(), effectiveQuantStrategy);
            return existing.key();
        }

        synchronized (loadedModels.lockFor(resolved)) {
            // Double-check inside lock
            existing = loadedModels.find(resolved);
            if (existing != null && existing.getQuantStrategy() == effectiveQuantStrategy) {
                return existing.key();
            }
            if (existing != null) {
                log.infof("DirectInferenceEngine: reloading [%s] for quantization strategy change %s -> %s",
                        resolved.getFileName(), existing.getQuantStrategy(), effectiveQuantStrategy);
                unloadModel(resolved);
            }

            log.infof("DirectInferenceEngine: loading model [%s] (strategy=%s)",
                    resolved.getFileName(), effectiveQuantStrategy);

            LoadedModel model = modelLoader().load(resolved, effectiveQuantStrategy);
            loadedModels.register(resolved, model);
            if (modelWarmupService != null) {
                runWithProfileSuspended(() -> modelWarmupService.warmUp(model));
            }

            log.infof("DirectInferenceEngine: loaded [%s] — %d weights, arch=%s",
                    model.key(), model.weights().size(), model.config().modelType());
            return model.key();
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
            DirectGenerationTimings timings = new DirectGenerationTimings();

            try {
                DirectRuntimePlatformBanner.print(metalBackend);

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                LoadedModel model = requireLoadedModel(modelPath, verbose, "[DEBUG]");

                if (verbose) {
                    System.out.println("[DEBUG] 3: get tokenizer/config");
                    System.out.flush();
                }
                Tokenizer tokenizer = model.tokenizer();
                ModelConfig config = model.config();
                if (verbose) {
                    System.out.println("[DEBUG] 4: architecture cache");
                    System.out.flush();
                }

                if (verbose) {
                    System.out.println("[DEBUG] 5: tokenize");
                    System.out.flush();
                }
                DirectPromptTokens promptTokens = DirectPromptTokens.encode(
                        tokenizer, config, model.runtimeTraits(), prompt, profile);
                long[] inputIds = promptTokens.ids();
                inputLen = promptTokens.length();
                if (verbose) {
                    System.out.printf("[DEBUG] 6: tokens=%d\n", inputLen);
                    System.out.flush();
                }
                if (verbose) {
                    promptTokens.debugSequence(tokenizer, "prompt");
                }

                Set<Integer> stops = requestStopTokenIds(model, cfg);

                try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                        .createSession(cfg.maxKvCacheTokens())) {
                    if (verbose) {
                        System.out.println("[DEBUG] 7: allocate session");
                        System.out.flush();
                    }
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                    if (verbose) {
                        System.out.println("[DEBUG] 8: prefill");
                        System.out.flush();
                    }
                    DirectGenerationExecutor.Result run = generationExecutor().runPrefill(
                            new DirectGenerationExecutor.PrefillRequest(
                                    model, cfg, session, inputIds, stops,
                                    DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                                    inputIds.length, requestStartNanos, profile, timings,
                                    false, true, false, null, null,
                                    verbose ? (token, step) -> debugChosenToken(tokenizer, token, step) : null));
                    if (verbose) {
                        System.out.println("[DEBUG] 9: prefill done");
                        System.out.flush();
                    }
                    if (verbose) {
                        debugChosenToken(tokenizer, run.prefill().token(), 0);
                    }

                    DirectGenerationLoop.Result loop = run.loop();
                    out.append(loop.text());
                    completionTokens = loop.completionTokens();
                    timings.recordLoop(loop);
                }

                return DirectInferenceResponses.finalBenchResponse(UUID.randomUUID().toString(), out.toString(),
                        modelPath, inputLen, completionTokens, t0, profile, backend,
                        timings.benchTimings(),
                        null);
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
        return DirectPromptTokens.encodeIds(model.tokenizer(), model.config(), model.runtimeTraits(), prompt, null);
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
            DirectGenerationTimings timings = new DirectGenerationTimings();

            try {
                DirectRuntimePlatformBanner.print(metalBackend);

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                LoadedModel model = requireLoadedModel(modelPath);
                ModelConfig config = model.config();

                DirectPromptTokens promptTokens = DirectPromptTokens.of(inputIds);
                inputLen = promptTokens.length();
                if (verbose) {
                    System.out.printf("[DEBUG] pretokenized tokens=%d%n", inputLen);
                    System.out.flush();
                }

                try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                        .createSession(cfg.maxKvCacheTokens())) {
                    long tAlloc0 = System.nanoTime();
                    session.allocate(config, cfg);
                    timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                    Set<Integer> stops = requestStopTokenIds(model, cfg);
                    DirectGenerationLoop.Result loop = generationExecutor().runPrefill(
                            new DirectGenerationExecutor.PrefillRequest(
                                    model, cfg, session, promptTokens.ids(), stops,
                                    DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                                    promptTokens.length(), requestStartNanos, profile, timings,
                                    true, false, false, null, null, null))
                            .loop();
                    out.append(loop.text());
                    generatedTokenIds = loop.generatedTokenIds();
                }

                return new DirectGenerationTrace(
                        DirectInferenceResponses.finalResponse(UUID.randomUUID().toString(), out.toString(),
                                modelPath, inputLen, generatedTokenIds.size(), t0, profile, backend,
                                Map.of("prompt_token_source", "pretokenized")),
                        promptTokens.ids().clone(),
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
            DirectGenerationTimings timings = new DirectGenerationTimings();

            try {
                DirectRuntimePlatformBanner.print(metalBackend);

                boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                LoadedModel model = requireLoadedModel(modelPath);
                ModelConfig config = model.config();

                DirectPromptTokens promptTokens = DirectPromptTokens.of(inputIds);
                inputLen = promptTokens.length();
                if (verbose) {
                    System.out.printf("[DEBUG] conversational pretokenized tokens=%d%n", inputLen);
                    System.out.flush();
                }

                session = getKVCacheManager().createSession(cfg.maxKvCacheTokens());
                long tAlloc0 = System.nanoTime();
                session.allocate(config, cfg);
                timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                Set<Integer> stops = requestStopTokenIds(model, cfg);
                DirectGenerationLoop.Result loop = generationExecutor().runPrefill(
                        new DirectGenerationExecutor.PrefillRequest(
                                model, cfg, session, promptTokens.ids(), stops,
                                DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                                promptTokens.length(), requestStartNanos, profile, timings,
                                true, false, false, null, null, null))
                        .loop();
                out.append(loop.text());
                generatedTokenIds = loop.generatedTokenIds();

                KVCacheManager.KVCacheSession retained = session;
                session = null;
                return new DirectConversationTrace(
                        DirectInferenceResponses.finalResponse(UUID.randomUUID().toString(), out.toString(),
                                modelPath, inputLen, generatedTokenIds.size(), t0, profile, backend,
                                Map.of("prompt_token_source", "pretokenized",
                                        "conversation_kv_retained", true)),
                        promptTokens.ids().clone(),
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
            int inputLen = fullInputIds == null ? 0 : fullInputIds.length;
            InferenceProfile profile = startProfile("sync");
            String backend = backendLabel(metalBackend);
            List<Long> generatedTokenIds = new ArrayList<>();
            DirectGenerationTimings timings = new DirectGenerationTimings();

            try {
                DirectRuntimePlatformBanner.print(metalBackend);

                if (session == null) {
                    throw new IllegalArgumentException("Conversation continuation requires an active KV cache session");
                }
                DirectConversationContinuationPlan continuation = DirectConversationContinuationPlan.resolve(
                        fullInputIds, cachedPrefixTokens, session.currentPos(), replayTokenId);

                LoadedModel model = requireLoadedModel(modelPath);

                Set<Integer> stops = requestStopTokenIds(model, cfg);
                DirectGenerationLoop.Result loop = generationExecutor().runContinuation(
                        new DirectGenerationExecutor.ContinuationRequest(
                                model, cfg, session, continuation, stops,
                                requestStartNanos, profile, timings,
                                true, false, false, null, null, null))
                        .loop();
                out.append(loop.text());
                generatedTokenIds = loop.generatedTokenIds();

                return new DirectConversationTrace(
                        DirectInferenceResponses.finalResponse(UUID.randomUUID().toString(), out.toString(),
                                modelPath, inputLen, generatedTokenIds.size(), t0, profile, backend,
                                continuation.retainedKvMetadata()),
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
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Streaming conversation generation failed", () -> clearProfile(), emitter -> {
                    Instant t0 = Instant.now();
                    long requestStartNanos = System.nanoTime();
                    String requestId = UUID.randomUUID().toString();
                    int inputLen = 0;
                    InferenceProfile profile = startProfile("stream");
                    String backend = backendLabel(metalBackend);
                    List<Long> generatedTokenIds = new ArrayList<>();
                    KVCacheManager.KVCacheSession session = null;
                    DirectGenerationTimings timings = new DirectGenerationTimings();

                    try {
                        DirectPromptTokens promptTokens = DirectPromptTokens.of(inputIds);
                        inputLen = promptTokens.length();
                        LoadedModel model = requireLoadedModel(modelPath);
                        ModelConfig config = model.config();

                        session = getKVCacheManager().createSession(cfg.maxKvCacheTokens());
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                        Set<Integer> stops = requestStopTokenIds(model, cfg);

                        final int streamInputLen = inputLen;
                        DirectGenerationLoop.Result loop = generationExecutor().runPrefill(
                                new DirectGenerationExecutor.PrefillRequest(
                                        model, cfg, session, promptTokens.ids(), stops,
                                        DirectGenerationStepSampler.SamplingMode.RAW_PRETOKENIZED,
                                        promptTokens.length(), requestStartNanos, profile, timings,
                                        true, false, true, emitter::isCancelled,
                                        delta -> emitter.emit(DirectInferenceResponses.streamDelta(requestId, delta,
                                                modelPath, streamInputLen, Map.of("prompt_token_source", "pretokenized",
                                                        "conversation_kv_retained", true))),
                                        null))
                                .loop();
                        generatedTokenIds = loop.generatedTokenIds();

                        InferenceResponse finalResponse = DirectInferenceResponses.finalResponse(requestId, "", modelPath,
                                inputLen, generatedTokenIds.size(), t0, profile, backend,
                                Map.of("prompt_token_source", "pretokenized",
                                        "conversation_kv_retained", true));
                        if (onComplete != null) {
                            onComplete.accept(new DirectConversationTrace(
                                    finalResponse,
                                    promptTokens.ids().clone(),
                                    toLongArray(generatedTokenIds),
                                    session));
                            session = null;
                        }
                        emitter.emit(finalResponse);

                    } catch (Throwable t) {
                        closeQuietly(session);
                        throw t;
                    }
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
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Streaming conversation continuation failed", () -> clearProfile(), emitter -> {
                    Instant t0 = Instant.now();
                    long requestStartNanos = System.nanoTime();
                    String requestId = UUID.randomUUID().toString();
                    int inputLen = fullInputIds == null ? 0 : fullInputIds.length;
                    InferenceProfile profile = startProfile("stream");
                    String backend = backendLabel(metalBackend);
                    List<Long> generatedTokenIds = new ArrayList<>();
                    DirectGenerationTimings timings = new DirectGenerationTimings();

                    if (session == null) {
                        throw new IllegalArgumentException(
                                "Conversation continuation requires an active KV cache session");
                    }
                    DirectConversationContinuationPlan continuation = DirectConversationContinuationPlan.resolve(
                            fullInputIds, cachedPrefixTokens, session.currentPos(), replayTokenId);

                    LoadedModel model = requireLoadedModel(modelPath);
                    Set<Integer> stops = requestStopTokenIds(model, cfg);
                    DirectGenerationLoop.Result loop = generationExecutor().runContinuation(
                            new DirectGenerationExecutor.ContinuationRequest(
                                    model, cfg, session, continuation, stops,
                                    requestStartNanos, profile, timings,
                                    true, false, true, emitter::isCancelled,
                                    delta -> emitter.emit(DirectInferenceResponses.streamDelta(requestId, delta,
                                            modelPath, inputLen, continuation.retainedKvMetadata())),
                                    null))
                            .loop();
                    generatedTokenIds = loop.generatedTokenIds();

                    InferenceResponse finalResponse = DirectInferenceResponses.finalResponse(requestId, "", modelPath,
                            inputLen, generatedTokenIds.size(), t0, profile, backend,
                            continuation.retainedKvMetadata());
                    if (onComplete != null) {
                        onComplete.accept(new DirectConversationTrace(
                                finalResponse,
                                fullInputIds.clone(),
                                toLongArray(generatedTokenIds),
                                session));
                    }
                    emitter.emit(finalResponse);
                });
    }

    public Multi<InferenceResponse> generateStream(String prompt, Path modelPath, GenerationConfig cfg) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Generation failed", () -> clearProfile(), emitter -> {
                    Instant t0 = Instant.now();
                    long requestStartNanos = System.nanoTime();
                    String requestId = UUID.randomUUID().toString();
                    int inputLen = 0;
                    InferenceProfile profile = startProfile("stream");
                    String backend = backendLabel(metalBackend);
                    DirectGenerationTimings timings = new DirectGenerationTimings();

                    boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
                    LoadedModel model = requireLoadedModel(modelPath, verbose, "[DEBUG-S]");

                    if (verbose) {
                        System.out.println("[DEBUG-S] 3: get tokenizer/config");
                        System.out.flush();
                    }
                    Tokenizer tokenizer = model.tokenizer();
                    ModelConfig config = model.config();
                    if (verbose) {
                        System.out.println("[DEBUG-S] 4: architecture cache");
                        System.out.flush();
                    }

                    if (verbose) {
                        System.out.println("[DEBUG-S] 5: tokenize");
                        System.out.printf("[DEBUG-PROMPT-TEXT] %s%n", DirectPromptTokens.printableText(prompt));
                        System.out.flush();
                    }
                    DirectPromptTokens promptTokens = DirectPromptTokens.encode(
                            tokenizer, config, model.runtimeTraits(), prompt, profile);
                    long[] inputIds = promptTokens.ids();
                    inputLen = promptTokens.length();
                    if (verbose) {
                        System.out.printf("[DEBUG-S] 6: tokens=%d\n", inputLen);
                        promptTokens.debugSequence(tokenizer, "prompt");
                        System.out.flush();
                    }

                    Set<Integer> stops = requestStopTokenIds(model, cfg);

                    int completionTokens = 0;
                    try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                            .createSession(cfg.maxKvCacheTokens())) {
                        if (verbose) {
                            System.out.println("[DEBUG-S] 7: allocate session");
                            System.out.flush();
                        }
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                        if (verbose) {
                            System.out.println("[DEBUG-S] 8: prefill");
                            System.out.flush();
                        }
                        final int streamInputLen = inputLen;
                        DirectGenerationExecutor.Result run = generationExecutor().runPrefill(
                                new DirectGenerationExecutor.PrefillRequest(
                                        model, cfg, session, inputIds, stops,
                                        DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                                        inputIds.length, requestStartNanos, profile, timings,
                                        false, true, true, emitter::isCancelled,
                                        delta -> emitter.emit(DirectInferenceResponses.streamDelta(requestId, delta,
                                                modelPath, streamInputLen, null)),
                                        null));
                        if (verbose) {
                            System.out.println("[DEBUG-S] 9: prefill done");
                            System.out.flush();
                        }

                        DirectGenerationLoop.Result loop = run.loop();
                        completionTokens = loop.completionTokens();
                        timings.recordLoop(loop);
                    }

                    emitter.emit(DirectInferenceResponses.finalBenchResponse(requestId, "", modelPath, inputLen,
                            completionTokens, t0, profile, backend,
                            timings.benchTimings(),
                            null));
                });
    }

    private static long[] toLongArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }

    public Multi<InferenceResponse> generateStream(long[] inputIds, Path modelPath, GenerationConfig cfg) {
        return DirectStreamExecution.create(STREAM_EXECUTOR_THREAD_FACTORY, log,
                "Generation failed", () -> clearProfile(), emitter -> {
                    Instant t0 = Instant.now();
                    long requestStartNanos = System.nanoTime();
                    String requestId = UUID.randomUUID().toString();
                    int inputLen = 0;
                    InferenceProfile profile = startProfile("stream");
                    String backend = backendLabel(metalBackend);
                    DirectGenerationTimings timings = new DirectGenerationTimings();

                    LoadedModel model = requireLoadedModel(modelPath);
                    ModelConfig config = model.config();

                    DirectPromptTokens promptTokens = DirectPromptTokens.of(inputIds);
                    inputLen = promptTokens.length();

                    int completionTokens = 0;
                    try (KVCacheManager.KVCacheSession session = getKVCacheManager()
                            .createSession(cfg.maxKvCacheTokens())) {
                        long tAlloc0 = System.nanoTime();
                        session.allocate(config, cfg);
                        timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);

                        Set<Integer> stops = requestStopTokenIds(model, cfg);

                        final int streamInputLen = inputLen;
                        DirectGenerationLoop.Result loop = generationExecutor().runPrefill(
                                new DirectGenerationExecutor.PrefillRequest(
                                        model, cfg, session, promptTokens.ids(), stops,
                                        DirectGenerationStepSampler.SamplingMode.TOKENIZER_AWARE,
                                        promptTokens.length(), requestStartNanos, profile, timings,
                                        false, true, true, emitter::isCancelled,
                                        delta -> emitter.emit(DirectInferenceResponses.streamDelta(requestId, delta,
                                                modelPath, streamInputLen,
                                                Map.of("prompt_token_source", "pretokenized"))),
                                        null))
                                .loop();
                        completionTokens = loop.completionTokens();
                        timings.recordLoop(loop);
                    }

                    emitter.emit(DirectInferenceResponses.finalBenchResponse(requestId, "", modelPath, inputLen,
                            completionTokens, t0, profile, backend,
                            timings.benchTimings(),
                            Map.of("prompt_token_source", "pretokenized")));

                });
    }

    @Override
    public void unloadModel(Path modelPath) {
        Path resolved = loadedModels.resolve(modelPath);
        LoadedModel model = loadedModels.remove(resolved);
        if (model != null) {
            DirectLoadedModelReleaser.release(model,
                    weights -> forwardPass().clearResolvedModelWeights(weights));
            log.infof("DirectInferenceEngine: unloaded [%s]", resolved.getFileName());
        }
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(Path modelPath) {
        return loadedModels.find(modelPath);
    }

    @Override
    public SafetensorEngine.LoadedModel getLoadedModel(String key) {
        return loadedModels.findByKey(key);
    }

    public boolean isLoaded(Path modelPath) {
        return loadedModels.contains(modelPath);
    }

    public Collection<LoadedModel> listLoadedModels() {
        return loadedModels.snapshot();
    }

    public QuantizationEngine getQuantizationEngine() {
        quantizationEngine = DirectInferenceDependencies.resolve(
                quantizationEngine, QuantizationEngine.class, "QuantizationEngine");
        return quantizationEngine;
    }

    public KVCacheManager getKVCacheManager() {
        kvCacheManager = DirectInferenceDependencies.resolve(kvCacheManager, KVCacheManager.class, "KVCacheManager");
        return kvCacheManager;
    }

    private LoadedModel requireLoadedModel(Path modelPath) {
        return requireLoadedModel(modelPath, false, null);
    }

    private LoadedModel requireLoadedModel(Path modelPath, boolean verbose, String debugPrefix) {
        return DirectLoadedModelAcquirer.require(
                () -> (LoadedModel) getLoadedModel(modelPath),
                () -> loadModel(modelPath),
                DirectInferenceProfiler::recordModelLoadNanos,
                verbose,
                debugPrefix);
    }

    private static Set<Integer> requestStopTokenIds(LoadedModel model, GenerationConfig cfg) {
        return GenerationTokenPolicy.mergeStopTokenIds(model.baseStopTokenIds(), cfg);
    }

}
