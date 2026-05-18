/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.metal.binding.MetalBinding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Full transformer forward pass using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectForwardPass {
    private static final Logger log = Logger.getLogger(DirectForwardPass.class);
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY = "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final String PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.prefer_native_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_bf16_to_f16_linear";
    private static final String DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_bf16_to_f16_linear";
    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long METAL_F16_WEIGHT_CACHE_MAX_BYTES = Long.getLong(
            METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES);
    private static final String LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.logits_large_half_cache_max_bytes";
    private static final long DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long LOGITS_LARGE_HALF_CACHE_MAX_BYTES = Long.getLong(
            LOGITS_LARGE_HALF_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_LOGITS_LARGE_HALF_CACHE_MAX_BYTES);
    private static final String FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_total_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES = 1536L * 1024L * 1024L;
    private static final long FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES = Long.getLong(
            FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES_PROPERTY,
            DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES);
    private static final String FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY =
            "gollek.safetensor.ffn_down_large_half_cache_per_tensor_max_bytes";
    private static final long DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES = 64L * 1024L * 1024L;
    private static final long FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES = Long.getLong(
            FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES_PROPERTY,
            DEFAULT_FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES);
    private static final boolean EXPERIMENTAL_METAL_LINEAR_ENABLED =
            resolveExperimentalMetalLinearEnabled();
    /** Gemma-4 Metal elementwise is opt-in until RMSNorm/add parity is proven. */
    private static final String ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_elementwise";
    private static final String DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_elementwise";
    private static final String METAL_ELEMENTWISE_MIN_SEQ_PROPERTY =
            "gollek.safetensor.metal_elementwise_min_seq";
    private static final String DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY =
            "gollek.safetensor.disable_gemma4_per_layer_input";
    private static final String DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_layer_scalar";
    private static final String ENABLE_GEMMA4_LAYER_SCALAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_layer_scalar";
    private static final String ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.allow_gemma4_fused_half_ffn";
    private static final String DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.disable_gemma4_fused_half_ffn";
    private static final String DISABLE_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_fused_ffn";
    private static final String ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_fused_ffn";
    private static final String ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY =
            "gollek.safetensor.enable_metal_fused_ffn_prefill";
    private static final String ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_matvec_ffn";
    private static final String ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_swiglu_matvec_ffn";
    private static final String DISABLE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_matvec_ffn";
    private static final String VALIDATE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.validate_metal_matvec_ffn";
    private static final String TRACE_FFN_FAST_PATH_PROPERTY =
            "gollek.safetensor.trace_ffn_fast_path";
    private static final Set<String> TRACED_FFN_FAST_PATH_DECISIONS = ConcurrentHashMap.newKeySet();
    private static final String REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY =
            "gollek.safetensor.reuse_ffn_projection_workspace";
    private static final String DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY =
            "gollek.safetensor.disable_ffn_projection_workspace_reuse";
    private static final String ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_linear_pair";
    private static final String DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_linear_pair";
    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final boolean AUTO_METAL_HALF_MATVEC_ENABLED =
            resolveAutoMetalHalfMatvecEnabled();
    private static final String ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec_pair";
    private static final String DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec_pair";
    private static final boolean DISABLE_METAL_HALF_MATVEC_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY);
    private static final boolean DISABLE_METAL_HALF_MATVEC_PAIR_ENABLED =
            Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY);
    private static final String ENABLE_METAL_HALF_MATVEC_VALUE =
            System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY);
    private static final String ENABLE_METAL_HALF_MATVEC_PAIR_VALUE =
            System.getProperty(ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY);
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 24576;
    private static final String ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_logits_mps_matvec";
    private static final String DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_logits_mps_matvec";
    private static final String METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_min_output";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT = 65536;
    private static final String METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY =
            "gollek.safetensor.metal_logits_mps_matvec_max_input";
    private static final int DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT = 4096;
    private static final String GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.gemma4_logits_metal_half_matvec_max_output";
    private static final int DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 300000;
    private static final String ENABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_post_ffn_norm";
    private static final String DISABLE_METAL_POST_FFN_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_post_ffn_norm";
    private static final boolean METAL_HALF_LINEAR_PAIR_ENABLED =
            resolveMetalHalfLinearPairEnabled();
    private static final String VERBOSE_TOKENS_PROPERTY = "gollek.verbose";
    private static final String VERBOSE_LAYERS_PROPERTY = "gollek.verbose.layers";
    private static final float GELU_INNER_SCALE = 0.79788456f;
    private static final float GELU_CUBIC_COEFF = 0.044715f;
    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;
    
    private MetalBinding metalBinding;
    private boolean metalReady;

    @jakarta.annotation.PostConstruct
    void init() {
        try {
            MetalBinding.initialize();
            this.metalBinding = MetalBinding.getInstance();
            this.metalBinding.init();
            String deviceName = this.metalBinding.deviceName();
            this.metalReady = this.metalBinding.isNativeAvailable()
                    && deviceName != null
                    && !deviceName.contains("CPU");
        } catch (Exception e) {
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.metalReady = false;
        }
    }

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(inputIds, weights, config, arch, kvCache);
        float[] result = materializeLogits(logits);
        logPrefillDiagnostics(result, config);
        return result;
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return prefill(embeddings, inputIds, null, weights, config, arch, kvCache);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache);
        return materializeLogits(logits);
    }

    public AccelTensor prefillLogitsTensor(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        AccelTensor embedded = embeddingLookup(embedTable, inputIds);
        float scale = resolveEmbeddingScale(arch, config, (int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        AccelTensor[] perLayerInputs = buildPerLayerInputs(inputIds, hidden, weights, config, arch);
        try {
            if (scale != 1.0f)
                embedded.close();
            return prefillLogitsTensor(hidden, inputIds, perLayerInputs, weights, config, arch, kvCache);
        } finally {
            closePerLayerInputs(perLayerInputs);
            hidden.closeWithParent();
        }
    }

    public AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        int startPos = kvCache.currentPos();
        long seqLen = embeddings.size(1);
        if (seqLen < 1) {
            throw new IllegalArgumentException("Invalid sequence length: " + seqLen + ". Prompt must result in at least one token.");
        }
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

        // Initial copy: embeddings -> hiddenASeg
        java.lang.foreign.MemorySegment.copy(embeddings.dataPtr(), 0, ws.getHiddenASeg(), 0, (long) seqLen * config.hiddenSize() * 4);

        java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
        java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

        boolean verboseTokens = Boolean.getBoolean(VERBOSE_TOKENS_PROPERTY);
        Map<Integer, FlashAttentionKernel.SharedKvState> sharedKvStates = null;
        if (config.resolvedNumKvSharedLayers() > 0) {
            if (startPos == 0) {
                kvCache.clearSharedKvStates();
            }
            sharedKvStates = kvCache.sharedKvStates();
        }
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            if (verboseTokens) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d start\n", i, config.numHiddenLayers());
                System.err.flush();
            }
            transformerLayer(currentHidden, nextHidden, inputIds, perLayerInputs != null ? perLayerInputs[i] : null,
                    weights, config, arch, kvCache, i, startPos, (int) seqLen, ws, sharedKvStates);

            // Swap buffers
            java.lang.foreign.MemorySegment temp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = temp;
        }

        AccelTensor hidden = AccelTensor.view(currentHidden, embeddings.shape());
        
        AccelTensor normed;
        if (canUseMetalElementwise(config, (int) seqLen)) {
            normed = AccelTensor.view(ws.getNormedAttnSeg(), hidden.shape());
            rmsNormRowsMetal(normed.dataPtr(), hidden.dataPtr(),
                    weights.get(arch.finalNormWeight()).dataPtr(), (int) seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                    useAddOneRmsNorm(arch, config));
        }
        if (hidden != embeddings)
            hidden.close();

        AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
        if (lmHeadW == null && config.tieWordEmbeddings()) {
            lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
        }
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        if (verboseTokens && seqLen > 1) {
            debugSequencePositionLogits(normed, lmHeadW, (int) seqLen);
        }
        AccelTensor lastPos = selectLastToken(normed, (int) seqLen);
        long tLogits0 = System.nanoTime();
        AccelTensor logits = linear(lastPos, lmHeadW, null, "logits", config);
        DirectInferenceEngine.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
        lastPos.closeWithParent();
        kvCache.advance((int) seqLen);
        return logits;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache);
        try {
            long tLogitsCopy0 = System.nanoTime();
            float[] result = logits.toFloatArray();
            DirectInferenceEngine.recordLogitsMaterializationNanos(System.nanoTime() - tLogitsCopy0);
            return result;
        } finally {
            logits.close();
        }
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache, false);
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache,
            boolean reuseLogitsOutput) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor embedded = embeddingLookup(embedTable, new long[] { tokenId });
        float scale = resolveEmbeddingScale(arch, config, (int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        AccelTensor[] perLayerInputs = buildPerLayerInputs(new long[] { tokenId }, hidden, weights, config, arch);
        try {
            if (scale != 1.0f)
                embedded.close();
            long[] tokenIds = new long[] { tokenId };

            KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
            ws.ensureCapacity(config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

            // Initial copy: hidden -> hiddenASeg
            java.lang.foreign.MemorySegment.copy(hidden.dataPtr(), 0, ws.getHiddenASeg(), 0, config.hiddenSize() * 4);
            hidden.close();

            java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
            java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

            Map<Integer, FlashAttentionKernel.SharedKvState> sharedKvStates = null;
            if (config.resolvedNumKvSharedLayers() > 0) {
                sharedKvStates = kvCache.sharedKvStates();
            }
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                transformerLayer(currentHidden, nextHidden, tokenIds, perLayerInputs != null ? perLayerInputs[i] : null,
                        weights, config, arch, kvCache, i, startPos, 1, ws, sharedKvStates);
                // Swap
                java.lang.foreign.MemorySegment temp = currentHidden;
                currentHidden = nextHidden;
                nextHidden = temp;
            }

            AccelTensor finalHidden = AccelTensor.view(currentHidden, new long[]{1, 1, config.hiddenSize()});
            
            AccelTensor normed;
            if (canUseMetalElementwise(config, 1)) {
                normed = AccelTensor.view(ws.getNormedAttnSeg(), finalHidden.shape());
                rmsNormRowsMetal(normed.dataPtr(), finalHidden.dataPtr(),
                        weights.get(arch.finalNormWeight()).dataPtr(), 1, config.hiddenSize(),
                        (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            } else {
                normed = AccelOps.rmsNorm(finalHidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                        useAddOneRmsNorm(arch, config));
            }
            finalHidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null && config.tieWordEmbeddings()) {
                lmHeadW = weights.get(arch.embedTokensWeight());
            }
            if (lmHeadW == null) {
                throw new IllegalStateException(
                        "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
            }
            AccelTensor logitsOutput = reuseLogitsOutput
                    ? reusableLogitsOutputTensor(ws, normed.shape(), lmHeadW)
                    : null;
            long tLogits0 = System.nanoTime();
            AccelTensor logits = linear(normed, lmHeadW, null, "logits", config, logitsOutput);
            DirectInferenceEngine.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
            if (normed.dataPtr() != ws.getNormedAttnSeg())
                normed.close();
            kvCache.advance(1);
            return logits;
        } finally {
            closePerLayerInputs(perLayerInputs);
            if (!hidden.isClosed())
                hidden.closeWithParent();
        }
    }

    private AccelTensor reusableLogitsOutputTensor(KVCacheManager.KVCacheSession.ForwardWorkspace ws,
                                                   long[] inputShape,
                                                   AccelTensor lmHeadW) {
        if (ws == null || inputShape == null || inputShape.length == 0 || lmHeadW == null || lmHeadW.shape().length != 2) {
            return null;
        }
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = lmHeadW.shape()[0];
        ws.ensureLogitsCapacity(elementCount(outputShape));
        return AccelTensor.view(ws.getLogitsSeg(), outputShape);
    }

    private float[] materializeLogits(AccelTensor logits) {
        try {
            long tLogitsCopy0 = System.nanoTime();
            float[] result = logits.toFloatArray();
            DirectInferenceEngine.recordLogitsMaterializationNanos(System.nanoTime() - tLogitsCopy0);
            return result;
        } finally {
            logits.close();
        }
    }

    private void logPrefillDiagnostics(float[] result, ModelConfig config) {
        if (!Boolean.getBoolean(VERBOSE_TOKENS_PROPERTY) || config.numAttentionHeads() <= 0) {
            return;
        }

        double sum = 0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float f : result) {
            sum += f;
            if (f < min) {
                min = f;
            }
            if (f > max) {
                max = f;
            }
        }

        System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d%n", min, max, sum, result.length);

        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < result.length; i++) {
            topIndices.add(i);
        }
        topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
        for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
            int id = topIndices.get(k);
            System.err.printf("  Top %d: ID=%d, val=%f%n", k, id, result[id]);
        }
    }

    private void transformerLayer(java.lang.foreign.MemorySegment hiddenIn, java.lang.foreign.MemorySegment hiddenOut, 
            long[] inputIds, AccelTensor perLayerInput, Map<String, AccelTensor> weights, ModelConfig config, 
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache, int layerIdx, int startPos, int seqLen, 
            KVCacheManager.KVCacheSession.ForwardWorkspace ws,
            Map<Integer, FlashAttentionKernel.SharedKvState> sharedKvStates) {
        
        long[] hiddenShape = new long[]{1, seqLen, config.hiddenSize()};
        boolean verboseLayers = Boolean.getBoolean(VERBOSE_LAYERS_PROPERTY);
        boolean useMetalElementwise = canUseMetalElementwise(config, seqLen);
        boolean useNativeElementwiseAdd = canUseNativeElementwiseAdd(config, seqLen);
        
        // Attention norm
        java.lang.foreign.MemorySegment normedAttnSeg = ws.getNormedAttnSeg();
        if (useMetalElementwise) {
            rmsNormRowsMetal(normedAttnSeg, hiddenIn,
                    weights.get(arch.layerAttentionNormWeight(layerIdx)).dataPtr(), seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            // Fallback (slow but stable)
            AccelTensor in = AccelTensor.view(hiddenIn, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, weights.get(arch.layerAttentionNormWeight(layerIdx)), config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedAttnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        FlashAttentionKernel.AttentionInput attnIn = new FlashAttentionKernel.AttentionInput(
                AccelTensor.view(normedAttnSeg, hiddenShape),
                weights.get(arch.layerQueryWeight(layerIdx)),
                weights.get(arch.layerKeyWeight(layerIdx)),
                weights.get(arch.layerValueWeight(layerIdx)),
                weights.get(arch.layerOutputWeight(layerIdx)),
                resolveBias(weights, arch.layerQueryBias(layerIdx)),
                resolveBias(weights, arch.layerKeyBias(layerIdx)),
                resolveBias(weights, arch.layerValueBias(layerIdx)),
                weights.get(arch.layerOutputBias(layerIdx)),
                arch, config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                weights.get(arch.layerQueryNormWeight(layerIdx)),
                weights.get(arch.layerKeyNormWeight(layerIdx)),
                weights.get(arch.layerPostAttnNormWeight(layerIdx)),
                sharedKvStates);

        if (verboseLayers) {
            System.err.printf("[DEBUG] Layer %d Attention start\n", layerIdx);
            System.err.flush();
        }
        long tAttention0 = System.nanoTime();
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        DirectInferenceEngine.recordAttentionNanos(System.nanoTime() - tAttention0);
        if (verboseLayers) {
            logTensorStats(attnOut, "layer " + layerIdx + " attnOut");
        }

        // First residual add after attention.
        residualAdd(hiddenIn, attnOut, hiddenOut, hiddenShape, seqLen, config.hiddenSize(), useNativeElementwiseAdd);
        attnOut.close();
        if (verboseLayers) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postAttnResidual");
        }

        // MLP input norm. Standard decoder-only blocks use post_attention_layernorm
        // before the FFN only; Gemma-style blocks may also expose a separate
        // post_feedforward_layernorm that is applied after the FFN output.
        AccelTensor preFfnNormW = weights.get(arch.layerPreFfnNormWeight(layerIdx));
        if (preFfnNormW == null) {
            preFfnNormW = weights.get(arch.layerFfnNormWeight(layerIdx));
        }
        AccelTensor postFfnNormW = weights.get(arch.layerPostFfnNormWeight(layerIdx));

        java.lang.foreign.MemorySegment normedFfnSeg = ws.getNormedFfnSeg();
        if (useMetalElementwise) {
            rmsNormRowsMetal(normedFfnSeg, hiddenOut, preFfnNormW.dataPtr(), seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            AccelTensor in = AccelTensor.view(hiddenOut, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, preFfnNormW, config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedFfnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        long tFfn0 = System.nanoTime();
        AccelTensor mlpOutputBuffer = AccelTensor.view(hiddenIn, hiddenShape);
        AccelTensor mlpOut = swigluFfn(AccelTensor.view(normedFfnSeg, hiddenShape), arch, config,
                weights.get(arch.layerFfnGateWeight(layerIdx)), weights.get(arch.layerFfnGateBias(layerIdx)),
                weights.get(arch.layerFfnUpWeight(layerIdx)), weights.get(arch.layerFfnUpBias(layerIdx)),
                weights.get(arch.layerFfnDownWeight(layerIdx)), weights.get(arch.layerFfnDownBias(layerIdx)), ws,
                mlpOutputBuffer);
        DirectInferenceEngine.recordFfnNanos(System.nanoTime() - tFfn0);
        if (verboseLayers) {
            logTensorStats(mlpOut, "layer " + layerIdx + " mlpOut");
        }

        // Post-FFN norm before the second residual add.
        AccelTensor mlpNormed;
        if (postFfnNormW != null) {
            if (useMetalElementwise && shouldUseMetalPostFfnNorm(config)) {
                rmsNormRowsMetal(normedFfnSeg, mlpOut.dataPtr(), postFfnNormW.dataPtr(), seqLen, config.hiddenSize(),
                        (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
                mlpNormed = AccelTensor.view(normedFfnSeg, hiddenShape);
            } else {
                mlpNormed = AccelOps.rmsNorm(mlpOut, postFfnNormW, config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            }
            mlpOut.close();
        } else {
            mlpNormed = mlpOut;
        }

        AccelTensor layerScalar = weights.get(arch.layerScalarWeight(layerIdx));
        residualAdd(hiddenOut, mlpNormed, hiddenOut, hiddenShape, seqLen, config.hiddenSize(), useNativeElementwiseAdd);
        mlpNormed.close();
        if (verboseLayers) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postFfnResidual");
        }

        applyPerLayerResidual(hiddenOut, hiddenShape, seqLen, config, arch, weights, layerIdx, perLayerInput);
        if (layerScalar != null && shouldApplyLayerScalar(config)) {
            float outputScale = readScalarValue(layerScalar);
            scaleSegmentInPlace(hiddenOut, hiddenShape, seqLen, config.hiddenSize(), outputScale);
        }
        if (verboseLayers && perLayerInput != null) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postPle");
        }
        if (verboseLayers && layerScalar != null) {
            float skipScale = readScalarValue(layerScalar);
            System.err.printf("[DEBUG] layer %d layerScalar=%f%n", layerIdx, skipScale);
        }
    }

    private void checkStability(AccelTensor t, String label) {
        float[] arr = t.toFloatArray();
        int nan = 0;
        int inf = 0;
        for (float f : arr) {
            if (Float.isNaN(f))
                nan++;
            if (Float.isInfinite(f))
                inf++;
        }
        if (nan > 0 || inf > 0) {
        }
    }

    private void logTensorStats(AccelTensor t, String label) {
        logArrayStats(t.toFloatArray(), label);
    }

    private void logSegmentStats(java.lang.foreign.MemorySegment seg, long[] shape, String label) {
        logTensorStats(AccelTensor.view(seg, shape), label);
    }

    private void logArrayStats(float[] arr, String label) {
        double sum = 0.0;
        double sumSq = 0.0;
        double sumAbs = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : arr) {
            sum += v;
            sumSq += (double) v * v;
            sumAbs += Math.abs(v);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double n = Math.max(1, arr.length);
        double mean = sum / n;
        double rms = Math.sqrt(sumSq / n);
        double meanAbs = sumAbs / n;
        System.err.printf("[DEBUG] %s stats: mean=%f meanAbs=%f rms=%f min=%f max=%f size=%d%n",
                label, mean, meanAbs, rms, min, max, arr.length);
        System.err.flush();
    }

    AccelTensor ffnNonGated(AccelTensor x, ModelConfig config, AccelTensor upW, AccelTensor downW) {
        AccelTensor up = linear(x, upW, null, "ffn_up_nongated", config);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = ffnDownLinear(act, downW, null, config, "ffn_down_nongated");
        act.close();
        return out;
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, KVCacheManager.KVCacheSession.ForwardWorkspace ws) {
        return swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, ws, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, ModelConfig config, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, KVCacheManager.KVCacheSession.ForwardWorkspace ws,
            AccelTensor downOutputBuffer) {
        AccelTensor metalMatvecFfn = tryMetalMatvecGatedFfn(
                x, arch, config, gateW, gateB, upW, upB, downW, downB, downOutputBuffer);
        if (metalMatvecFfn != null) {
            return metalMatvecFfn;
        }

        AccelTensor metalFused = tryMetalFusedGatedFfn(
                x, arch, config, gateW, gateB, upW, upB, downW, downB, downOutputBuffer);
        if (metalFused != null) {
            return metalFused;
        }

        AccelTensor combinedBuffer = null;
        if (ws != null && gateW != null && upW != null
                && gateW.shape().length == 2
                && upW.shape().length == 2
                && gateW.shape()[0] == upW.shape()[0]) {
            ws.ensureCapacity(x.numel(), x.size(-1), gateW.shape()[0]);
            long requiredCombinedBytes = x.size(0) * x.size(1) * gateW.shape()[0] * Float.BYTES;
            if (ws.getCombinedSeg() != null && ws.getCombinedSeg().byteSize() >= requiredCombinedBytes) {
                combinedBuffer = AccelTensor.view(ws.getCombinedSeg(), new long[] { x.size(0), x.size(1), gateW.shape()[0] });
            }
        }
        boolean preferSeparateMetalHalf = shouldPreferSeparateMetalHalfFfn(x, gateW, upW, config);
        boolean allowFusedHalfFfn = !isGemma4Text(config)
                || Boolean.getBoolean(ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY);
        AccelTensor fusedCombined = (preferSeparateMetalHalf || !allowFusedHalfFfn)
                ? null
                : tryFusedGatedHalfFfn(x, gateW, gateB, upW, upB,
                        arch.activationType() == FFNActivationType.GELU, combinedBuffer);
        if (fusedCombined != null) {
            AccelTensor out = ffnDownLinear(fusedCombined, downW, downB, config, "ffn_down", downOutputBuffer);
            if (ws == null || fusedCombined.dataPtr() != ws.getCombinedSeg()) {
                fusedCombined.close();
            }
            return out;
        }

        long[] gateShape = new long[] { x.size(0), x.size(1), gateW.shape()[0] };
        boolean reuseFfnProjectionWorkspace = canReuseFfnProjectionWorkspace();
        if (reuseFfnProjectionWorkspace && ws != null) {
            ws.ensureProjectionScratchCapacity(elementCount(gateShape) * Float.BYTES);
        }
        AccelTensor gateBuffer = reuseFfnProjectionWorkspace
                ? reusableWorkspaceView(ws == null ? null : ws.getGateSeg(), gateShape)
                : null;
        AccelTensor upBuffer = reuseFfnProjectionWorkspace
                ? reusableWorkspaceView(ws == null ? null : ws.getUpSeg(), gateShape)
                : null;

        LinearPair pairedGateUp = tryMetalHalfLinearPair(x, config, gateW, gateB, upW, upB, gateBuffer, upBuffer);
        AccelTensor gate;
        AccelTensor up;
        if (pairedGateUp != null) {
            gate = pairedGateUp.first();
            up = pairedGateUp.second();
        } else {
            if (reuseFfnProjectionWorkspace && ws != null) {
                gateBuffer = reusableWorkspaceView(ws.getGateSeg(), gateShape);
                upBuffer = reusableWorkspaceView(ws.getUpSeg(), gateShape);
            }
            gate = linear(x, gateW, gateB, "ffn_gate", config, gateBuffer);
            up = linear(x, upW, upB, "ffn_up", config, upBuffer);
        }

        AccelTensor combined;
        if (arch.activationType() == FFNActivationType.GELU) {
            if (ws != null) {
                combined = AccelTensor.view(ws.getCombinedSeg(), gate.shape());
                if (canUseMetalElementwise(config, (int) x.size(1))) {
                    try {
                        int rc = metalBinding.geluFfn(combined.dataPtr(), gate.dataPtr(), up.dataPtr(),
                                (int) gate.numel());
                        if (rc != 0) {
                            throw new IllegalStateException("Metal geluFfn failed with code " + rc);
                        }
                    } catch (IllegalStateException | UnsupportedOperationException e) {
                        fusedGeglu(combined, gate, up);
                    }
                } else {
                    fusedGeglu(combined, gate, up);
                }
            } else {
                AccelTensor activated = AccelOps.gelu(gate);
                gate.close();
                combined = AccelOps.mul(activated, up);
                activated.close();
            }
        } else if (canUseMetalElementwise(config, (int) x.size(1)) && ws != null) {
            combined = AccelTensor.view(ws.getCombinedSeg(), gate.shape());
            try {
                int rc = metalBinding.siluFfn(combined.dataPtr(), gate.dataPtr(), up.dataPtr(),
                        (int) gate.numel());
                if (rc != 0) {
                    throw new IllegalStateException("Metal siluFfn failed with code " + rc);
                }
            } catch (IllegalStateException | UnsupportedOperationException e) {
                combined = AccelOps.swiglu(gate, up);
            }
        } else {
            combined = AccelOps.swiglu(gate, up);
        }

        if (!gate.isClosed()) gate.close();
        up.close();
        AccelTensor out = ffnDownLinear(combined, downW, downB, config, "ffn_down", downOutputBuffer);
        if (ws == null || combined.dataPtr() != ws.getCombinedSeg()) {
            combined.close();
        }
        return out;
    }

    private AccelTensor tryMetalFusedGatedFfn(AccelTensor input,
                                               ModelArchitecture arch,
                                               ModelConfig config,
                                               AccelTensor gateW,
                                               AccelTensor gateB,
                                               AccelTensor upW,
                                               AccelTensor upB,
        AccelTensor downW,
        AccelTensor downB,
        AccelTensor outputBuffer) {
        boolean siluActivation = arch.activationType() == FFNActivationType.SILU;
        boolean geluActivation = arch.activationType() == FFNActivationType.GELU;
        if (Boolean.getBoolean(DISABLE_METAL_FUSED_FFN_PROPERTY)) {
            return rejectMetalFusedGatedFfn("disabled", config, input, gateW, upW, downW);
        }
        if (!siluActivation && !geluActivation) {
            return rejectMetalFusedGatedFfn("unsupported_activation:" + arch.activationType(),
                    config, input, gateW, upW, downW);
        }
        boolean gemma4Text = isGemma4Text(config);
        if (geluActivation
                && !gemma4Text
                && !Boolean.getBoolean(ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY)) {
            return rejectMetalFusedGatedFfn("geglu_flag_disabled", config, input, gateW, upW, downW);
        }
        if (gateB != null || upB != null || downB != null) {
            return rejectMetalFusedGatedFfn("bias_present", config, input, gateW, upW, downW);
        }
        if (!canUseExperimentalMetalLinear()) {
            return rejectMetalFusedGatedFfn("metal_linear_disabled", config, input, gateW, upW, downW);
        }
        if (metalBinding == null) {
            return rejectMetalFusedGatedFfn("metal_binding_unavailable", config, input, gateW, upW, downW);
        }
        if (siluActivation && !metalBinding.supportsSwigluFfnHalf()) {
            return rejectMetalFusedGatedFfn("swiglu_symbol_unavailable", config, input, gateW, upW, downW);
        }
        if (geluActivation && !metalBinding.supportsGegluFfnHalf()) {
            return rejectMetalFusedGatedFfn("geglu_symbol_unavailable", config, input, gateW, upW, downW);
        }
        if (!canUseMetalHalfLinearCandidate(input, gateW, config)) {
            return rejectMetalFusedGatedFfn("gate_candidate_ineligible", config, input, gateW, upW, downW);
        }
        if (!canUseMetalHalfLinearCandidate(input, upW, config)) {
            return rejectMetalFusedGatedFfn("up_candidate_ineligible", config, input, gateW, upW, downW);
        }
        if (!canUseMetalHalfWeight(downW, config)) {
            return rejectMetalFusedGatedFfn("down_weight_ineligible", config, input, gateW, upW, downW);
        }

        long[] gateShape = gateW.shape();
        long[] upShape = upW.shape();
        long[] downShape = downW.shape();
        if (gateShape.length != 2
                || upShape.length != 2
                || downShape.length != 2
                || gateShape[0] != upShape[0]
                || gateShape[1] != upShape[1]
                || downShape[1] != gateShape[0]) {
            return rejectMetalFusedGatedFfn("shape_mismatch", config, input, gateW, upW, downW);
        }

        long[] inputShape = input.shape();
        long inputDim = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows <= 0L) {
            return rejectMetalFusedGatedFfn("invalid_rows:" + rows, config, input, gateW, upW, downW);
        }
        if (gemma4Text && !allowGemma4FusedHalfFfn(rows, config)) {
            return rejectMetalFusedGatedFfn(rows == 1L
                    ? "gemma4_decode_flag_disabled"
                    : "gemma4_prefill_flag_disabled",
                    config, input, gateW, upW, downW);
        }
        if (rows != 1L && !shouldUseMetalFusedFfnPrefill(config)) {
            return rejectMetalFusedGatedFfn("prefill_flag_disabled:rows=" + rows, config, input, gateW, upW, downW);
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(gateW, config)
                && shouldUseNativeMetalBf16Linear(upW, config)
                && shouldUseNativeMetalBf16Linear(downW, config);
        AccelTensor metalGateW = toMetalHalfWeight(gateW, nativeBf16Weights, config);
        AccelTensor metalUpW = toMetalHalfWeight(upW, nativeBf16Weights, config);
        AccelTensor metalDownW = toMetalHalfWeight(downW, nativeBf16Weights, config);
        if (metalGateW == null || metalUpW == null || metalDownW == null) {
            return rejectMetalFusedGatedFfn("weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long intermediateDim = metalGateW.shape()[0];
        long outputDim = metalDownW.shape()[0];
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = outputDim;
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int rc;
            if (siluActivation) {
                rc = metalBinding.swigluFfnHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(rows),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim),
                        nativeBf16Weights);
            } else {
                rc = metalBinding.gegluFfnHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(rows),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim),
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal fused gated FFN failed with code " + rc);
            }
            DirectInferenceEngine.recordLinearNanos(
                    siluActivation ? "ffn_fused_metal" : "ffn_geglu_fused_metal",
                    System.nanoTime() - t0);
            traceFfnFastPath("fused-gated-ffn", "accept:"
                            + (siluActivation ? "swiglu" : "geglu")
                            + ":native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            traceFfnFastPath("fused-gated-ffn", "reject:runtime_failure:" + e.getClass().getSimpleName(),
                    config, input, gateW, upW, downW);
            log.debugf("Falling back from fused Metal gated FFN: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor tryMetalMatvecGatedFfn(AccelTensor input,
                                               ModelArchitecture arch,
                                               ModelConfig config,
                                               AccelTensor gateW,
                                               AccelTensor gateB,
                                               AccelTensor upW,
                                               AccelTensor upB,
                                               AccelTensor downW,
                                               AccelTensor downB,
                                               AccelTensor outputBuffer) {
        boolean siluActivation = arch.activationType() == FFNActivationType.SILU;
        boolean geluActivation = arch.activationType() == FFNActivationType.GELU;
        if (geluActivation && !shouldUseMetalGegluMatvecFfn(config)) {
            traceFfnFastPath("matvec-gated-ffn", "reject:geglu_flag_disabled", config, input, gateW, upW, downW);
            return null;
        }
        if (siluActivation && !shouldUseMetalSwigluMatvecFfn(config)) {
            traceFfnFastPath("matvec-gated-ffn", "reject:swiglu_flag_disabled", config, input, gateW, upW, downW);
            return null;
        }
        if (!siluActivation && !geluActivation) {
            traceFfnFastPath("matvec-gated-ffn", "reject:unsupported_activation:" + arch.activationType(),
                    config, input, gateW, upW, downW);
            return null;
        }
        if (gateB != null || upB != null || downB != null) {
            traceFfnFastPath("matvec-gated-ffn", "reject:bias_present", config, input, gateW, upW, downW);
            return null;
        }
        if (!canUseExperimentalMetalLinear() || metalBinding == null) {
            traceFfnFastPath("matvec-gated-ffn", "reject:metal_unavailable", config, input, gateW, upW, downW);
            return null;
        }
        if (!canUseMetalHalfLinearCandidate(input, gateW, config)
                || !canUseMetalHalfLinearCandidate(input, upW, config)
                || !canUseMetalHalfWeight(downW, config)) {
            traceFfnFastPath("matvec-gated-ffn", "reject:candidate_ineligible",
                    config, input, gateW, upW, downW);
            return null;
        }

        long[] gateShape = gateW.shape();
        long[] upShape = upW.shape();
        long[] downShape = downW.shape();
        if (gateShape.length != 2
                || upShape.length != 2
                || downShape.length != 2
                || gateShape[0] != upShape[0]
                || gateShape[1] != upShape[1]
                || downShape[1] != gateShape[0]) {
            traceFfnFastPath("matvec-gated-ffn", "reject:shape_mismatch", config, input, gateW, upW, downW);
            return null;
        }

        long[] inputShape = input.shape();
        long inputDim = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, inputDim);
        if (rows != 1L) {
            traceFfnFastPath("matvec-gated-ffn", "reject:not_single_token_rows:" + rows,
                    config, input, gateW, upW, downW);
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(gateW, config)
                && shouldUseNativeMetalBf16Linear(upW, config)
                && shouldUseNativeMetalBf16Linear(downW, config);
        if (nativeBf16Weights && siluActivation && !metalBinding.supportsSwigluFfnMatvecBf16()) {
            traceFfnFastPath("matvec-gated-ffn", "reject:swiglu_bf16_matvec_symbol_unavailable",
                    config, input, gateW, upW, downW);
            return null;
        }
        if (nativeBf16Weights && geluActivation && !metalBinding.supportsGegluFfnMatvecBf16()) {
            traceFfnFastPath("matvec-gated-ffn", "reject:geglu_bf16_matvec_symbol_unavailable",
                    config, input, gateW, upW, downW);
            return null;
        }
        if (!nativeBf16Weights && siluActivation && !metalBinding.supportsSwigluFfnMatvecHalf()) {
            traceFfnFastPath("matvec-gated-ffn", "reject:swiglu_matvec_symbol_unavailable",
                    config, input, gateW, upW, downW);
            return null;
        }
        if (!nativeBf16Weights && geluActivation && !metalBinding.supportsGegluFfnMatvecHalf()) {
            traceFfnFastPath("matvec-gated-ffn", "reject:geglu_matvec_symbol_unavailable",
                    config, input, gateW, upW, downW);
            return null;
        }

        AccelTensor metalGateW = toMetalHalfWeight(gateW, nativeBf16Weights, config);
        AccelTensor metalUpW = toMetalHalfWeight(upW, nativeBf16Weights, config);
        AccelTensor metalDownW = toMetalHalfWeight(downW, nativeBf16Weights, config);
        if (metalGateW == null || metalUpW == null || metalDownW == null) {
            traceFfnFastPath("matvec-gated-ffn", "reject:weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return null;
        }
        AccelTensor.QuantType weightType = metalGateW.quantType();
        if (metalUpW.quantType() != weightType
                || metalDownW.quantType() != weightType
                || (weightType != AccelTensor.QuantType.F16 && weightType != AccelTensor.QuantType.BF16)) {
            traceFfnFastPath("matvec-gated-ffn", "reject:weight_type_mismatch:native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long intermediateDim = metalGateW.shape()[0];
        long outputDim = metalDownW.shape()[0];
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = outputDim;
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            int rc;
            if (nativeBf16Weights && siluActivation) {
                rc = metalBinding.swigluFfnMatvecBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else if (nativeBf16Weights) {
                rc = metalBinding.gegluFfnMatvecBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else if (siluActivation) {
                rc = metalBinding.swigluFfnMatvecHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            } else {
                rc = metalBinding.gegluFfnMatvecHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalGateW.dataPtr(),
                        metalUpW.dataPtr(),
                        metalDownW.dataPtr(),
                        Math.toIntExact(inputDim),
                        Math.toIntExact(intermediateDim),
                        Math.toIntExact(outputDim));
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matvec gated FFN failed with code " + rc);
            }
            if (shouldValidateMetalMatvecFfn() && !allFinite(out)) {
                throw new IllegalStateException("Metal matvec gated FFN produced non-finite output");
            }
            DirectInferenceEngine.recordLinearNanos(
                    nativeBf16Weights
                            ? (siluActivation ? "ffn_swiglu_matvec_bf16" : "ffn_geglu_matvec_bf16")
                            : (siluActivation ? "ffn_swiglu_matvec_metal" : "ffn_geglu_matvec_metal"),
                    System.nanoTime() - t0);
            traceFfnFastPath("matvec-gated-ffn",
                    "accept:" + (siluActivation ? "swiglu" : "geglu") + ":native_bf16=" + nativeBf16Weights,
                    config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            traceFfnFastPath("matvec-gated-ffn", "reject:runtime_failure:" + e.getClass().getSimpleName(),
                    config, input, gateW, upW, downW);
            log.debugf("Falling back from Metal matvec gated FFN: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor rejectMetalFusedGatedFfn(String reason,
                                                 ModelConfig config,
                                                 AccelTensor input,
                                                 AccelTensor gateW,
                                                 AccelTensor upW,
                                                 AccelTensor downW) {
        traceFfnFastPath("fused-gated-ffn", "reject:" + reason, config, input, gateW, upW, downW);
        return null;
    }

    private void traceFfnFastPath(String path,
                                  String decision,
                                  ModelConfig config,
                                  AccelTensor input,
                                  AccelTensor gateW,
                                  AccelTensor upW,
                                  AccelTensor downW) {
        if (!Boolean.getBoolean(TRACE_FFN_FAST_PATH_PROPERTY)) {
            return;
        }
        String key = path + "|" + decision + "|" + tensorSummary(input)
                + "|" + tensorSummary(gateW) + "|" + tensorSummary(upW) + "|" + tensorSummary(downW);
        if (!TRACED_FFN_FAST_PATH_DECISIONS.add(key)) {
            return;
        }
        System.err.printf("[gollek-ffn] path=%s decision=%s model=%s input=%s gate=%s up=%s down=%s%n",
                path,
                decision,
                config != null ? config.modelType() : "unknown",
                tensorSummary(input),
                tensorSummary(gateW),
                tensorSummary(upW),
                tensorSummary(downW));
    }

    private static String tensorSummary(AccelTensor tensor) {
        if (tensor == null) {
            return "null";
        }
        return tensor.quantType() + Arrays.toString(tensor.shape());
    }

    private static boolean allFinite(AccelTensor tensor) {
        if (tensor == null || tensor.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        MemorySegment segment = tensor.dataPtr();
        long n = tensor.numel();
        for (long i = 0; i < n; i++) {
            if (!Float.isFinite(segment.getAtIndex(ValueLayout.JAVA_FLOAT, i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean shouldValidateMetalMatvecFfn() {
        return Boolean.getBoolean(VALIDATE_METAL_MATVEC_FFN_PROPERTY)
                || Boolean.getBoolean(TRACE_FFN_FAST_PATH_PROPERTY);
    }

    private boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfig config) {
        if (weight == null || weight.shape().length != 2 || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && isGemma4Text(config)) {
            return shouldUseNativeMetalBf16Linear(weight, config)
                    || allowGemma4Bf16ToF16Linear(config);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(config));
    }

    private AccelTensor toMetalHalfWeight(AccelTensor weight, boolean nativeBf16, ModelConfig config) {
        if (weight == null) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.F16) {
            return weight;
        }
        if (nativeBf16 && weight.quantType() == AccelTensor.QuantType.BF16) {
            return weight;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && isGemma4Text(config)) {
            return allowGemma4Bf16ToF16Linear(config)
                    ? weight.toF16CachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES)
                    : null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(config)) {
            return weight.toF16CachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
        }
        return null;
    }

    private boolean shouldPreferSeparateMetalHalfFfn(AccelTensor input, AccelTensor gateW, AccelTensor upW,
            ModelConfig config) {
        return canUseMetalHalfLinearCandidate(input, gateW, config)
                && canUseMetalHalfLinearCandidate(input, upW, config);
    }

    private static boolean allowGemma4FusedHalfFfn() {
        if (Boolean.getBoolean(DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return false;
    }

    private boolean allowGemma4FusedHalfFfn(long rows, ModelConfig config) {
        if (!isGemma4Text(config)) {
            return true;
        }
        if (Boolean.getBoolean(DISABLE_GEMMA4_FUSED_HALF_FFN_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ALLOW_GEMMA4_FUSED_HALF_FFN_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        String prefillExplicit = System.getProperty(ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY);
        if (prefillExplicit != null && !prefillExplicit.isBlank()) {
            return rows > 1L && Boolean.parseBoolean(prefillExplicit);
        }
        return rows > 1L && shouldUseMetalFusedFfnPrefill(config);
    }

    private boolean shouldUseMetalFusedFfnPrefill(ModelConfig config) {
        String explicit = System.getProperty(ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // Keep opt-in for now: on M4, the separate Metal pair/down path is
        // faster for Gemma-4 BF16 prefill than the fused command buffer.
        return false;
    }

    private boolean shouldUseMetalGegluMatvecFfn(ModelConfig config) {
        if (Boolean.getBoolean(DISABLE_METAL_MATVEC_FFN_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // The native BF16 matvec FFN is available for experiments, but the
        // separate gate/up pair plus down projection wins on current M4 runs.
        return false;
    }

    private boolean shouldUseMetalSwigluMatvecFfn(ModelConfig config) {
        if (Boolean.getBoolean(DISABLE_METAL_MATVEC_FFN_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return false;
    }

    private LinearPair tryMetalHalfLinearPair(AccelTensor input,
                                              ModelConfig config,
                                              AccelTensor firstWeight,
                                              AccelTensor firstBias,
                                              AccelTensor secondWeight,
                                              AccelTensor secondBias,
                                              AccelTensor firstOutputBuffer,
                                              AccelTensor secondOutputBuffer) {
        if (!shouldUseMetalHalfLinearPair(config, input)) {
            traceFfnFastPath("gate-up-pair", "reject:disabled", config, input, firstWeight, secondWeight, null);
            return null;
        }
        if (!canUseMetalHalfLinearCandidate(input, firstWeight, config)) {
            traceFfnFastPath("gate-up-pair", "reject:first_candidate_ineligible",
                    config, input, firstWeight, secondWeight, null);
            return null;
        }
        if (!canUseMetalHalfLinearCandidate(input, secondWeight, config)) {
            traceFfnFastPath("gate-up-pair", "reject:second_candidate_ineligible",
                    config, input, firstWeight, secondWeight, null);
            return null;
        }
        if (!metalBinding.supportsMatmulTransposedRightHalfPair()) {
            traceFfnFastPath("gate-up-pair", "reject:pair_symbol_unavailable",
                    config, input, firstWeight, secondWeight, null);
            return null;
        }
        if (firstWeight.shape()[0] != secondWeight.shape()[0]
                || firstWeight.shape()[1] != secondWeight.shape()[1]) {
            traceFfnFastPath("gate-up-pair", "reject:shape_mismatch",
                    config, input, firstWeight, secondWeight, null);
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(firstWeight, config)
                && shouldUseNativeMetalBf16Linear(secondWeight, config);
        AccelTensor metalFirstWeight = toMetalHalfWeight(firstWeight, nativeBf16Weights, config);
        AccelTensor metalSecondWeight = toMetalHalfWeight(secondWeight, nativeBf16Weights, config);
        if (metalFirstWeight == null || metalSecondWeight == null
                || metalFirstWeight.quantType() != metalSecondWeight.quantType()
                || (metalFirstWeight.quantType() != AccelTensor.QuantType.F16
                        && metalFirstWeight.quantType() != AccelTensor.QuantType.BF16)) {
            traceFfnFastPath("gate-up-pair", "reject:weight_conversion_failed:native_bf16=" + nativeBf16Weights,
                    config, input, firstWeight, secondWeight, null);
            return null;
        }

        long t0 = System.nanoTime();
        long[] inputShape = input.shape();
        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = metalFirstWeight.shape()[0];
        AccelTensor first = reusableOutputTensor(firstOutputBuffer, outputShape);
        AccelTensor second = reusableOutputTensor(secondOutputBuffer, outputShape);

        try {
            long k = inputShape[inputShape.length - 1];
            long rows = input.numel() / Math.max(1L, k);
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(metalFirstWeight.shape()[0]);
            int rc = -2;
            String executionPath = "mps";
            if (m == 1
                    && nativeBf16Weights
                    && shouldUseMetalHalfMatvecPair(config, n)
                    && metalBinding.supportsMatvecTransposedRightBf16Pair()) {
                rc = metalBinding.matvecTransposedRightBf16Pair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "bf16_matvec";
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weights
                    && shouldUseMetalHalfMatvecPair(config, n)
                    && metalBinding.supportsMatvecTransposedRightHalfPair()) {
                rc = metalBinding.matvecTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "matvec";
                }
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPair failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceEngine.recordLinearNanos("ffn_gate_up_pair", System.nanoTime() - t0);
            traceFfnFastPath("gate-up-pair", "accept:" + executionPath + ":native_bf16=" + nativeBf16Weights,
                    config, input, firstWeight, secondWeight, null);
            return new LinearPair(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            traceFfnFastPath("gate-up-pair", "reject:runtime_failure:" + e.getClass().getSimpleName(),
                    config, input, firstWeight, secondWeight, null);
            log.debugf("Falling back from Metal half linear pair to separate linears: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static boolean resolveMetalHalfLinearPairEnabled() {
        if (Boolean.getBoolean(DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    private static boolean resolveAutoMetalHalfMatvecEnabled() {
        String explicit = System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    private boolean shouldUseMetalHalfLinearPair(ModelConfig config, AccelTensor input) {
        if (!METAL_HALF_LINEAR_PAIR_ENABLED) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        if (isGemma4Text(config)) {
            return isMultiRowLinearInput(input) || !allowGemma4FusedHalfFfn();
        }
        return !hasGemma4StylePerLayerInputs(config);
    }

    private static boolean isMultiRowLinearInput(AccelTensor input) {
        if (input == null || input.shape().length == 0) {
            return false;
        }
        long hidden = input.shape()[input.shape().length - 1];
        if (hidden <= 0L) {
            return false;
        }
        return input.numel() / hidden > 1L;
    }

    private AccelTensor addBiasIfNeeded(AccelTensor tensor, AccelTensor bias) {
        if (bias == null) {
            return tensor;
        }
        AccelTensor biased = AccelOps.add(tensor, bias);
        tensor.close();
        return biased;
    }

    private record LinearPair(AccelTensor first, AccelTensor second) {
    }

    private AccelTensor tryFusedGatedHalfFfn(AccelTensor x,
                                             AccelTensor gateW,
                                             AccelTensor gateB,
                                             AccelTensor upW,
                                             AccelTensor upB,
                                             boolean geluActivation,
                                             AccelTensor outputBuffer) {
        if (gateW == null || upW == null || gateW.shape().length != 2 || upW.shape().length != 2) {
            return null;
        }
        if (gateW.shape()[0] != upW.shape()[0]) {
            return null;
        }
        long[] outputShape = new long[] { x.size(0), x.size(1), gateW.shape()[0] };
        AccelTensor buffer = outputBuffer;
        if (buffer != null) {
            long[] bufferShape = buffer.shape();
            if (bufferShape.length != outputShape.length) {
                buffer = null;
            } else {
                for (int i = 0; i < outputShape.length; i++) {
                    if (bufferShape[i] != outputShape[i]) {
                        buffer = null;
                        break;
                    }
                }
            }
        }
        long t0 = System.nanoTime();
        AccelTensor combined = invokeFusedGatedHalfLinear(
                x, gateW, gateB, upW, upB, geluActivation, buffer);
        if (combined != null) {
            DirectInferenceEngine.recordLinearNanos("ffn_gate_up_fused", System.nanoTime() - t0);
        }
        return combined;
    }

    private AccelTensor invokeFusedGatedHalfLinear(AccelTensor x,
                                                   AccelTensor gateW,
                                                   AccelTensor gateB,
                                                   AccelTensor upW,
                                                   AccelTensor upB,
                                                   boolean geluActivation,
                                                   AccelTensor outputBuffer) {
        return AccelOps.fusedGatedHalfLinear(x, gateW, gateB, upW, upB, geluActivation, outputBuffer);
    }

    /**
     * In-place GeGLU combine for Gemma-style FFNs.
     * Writes gelu(gate) * up directly into the reusable workspace buffer.
     */
    private void fusedGeglu(AccelTensor out, AccelTensor gate, AccelTensor up) {
        java.lang.foreign.MemorySegment outSeg = out.dataPtr();
        java.lang.foreign.MemorySegment gateSeg = gate.dataPtr();
        java.lang.foreign.MemorySegment upSeg = up.dataPtr();
        long n = gate.numel();

        for (long i = 0; i < n; i++) {
            float g = gateSeg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float u = upSeg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float inner = GELU_INNER_SCALE * (g + GELU_CUBIC_COEFF * g * g * g);
            float gelu = 0.5f * g * (1.0f + (float) Math.tanh(inner));
            outSeg.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, gelu * u);
        }
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        int seqLen = tokenIds.length;
        AccelTensor selected = embedTable.indexSelect(tokenIds);
        long hiddenSize = selected.size(1);
        return selected.reshape(1L, seqLen, hiddenSize);
    }

    private AccelTensor selectLastToken(AccelTensor hidden, int seqLen) {
        if (seqLen < 1) return hidden; // Safety fallback
        long hiddenSize = hidden.size(hidden.shape().length - 1);
        return hidden.slice(1, seqLen - 1L, seqLen).reshape(1L, hiddenSize);
    }

    private AccelTensor selectTokenAt(AccelTensor hidden, int tokenIndex) {
        long hiddenSize = hidden.size(hidden.shape().length - 1);
        long clamped = Math.max(0L, tokenIndex);
        return hidden.slice(1, clamped, clamped + 1L).reshape(1L, hiddenSize);
    }

    private void debugSequencePositionLogits(AccelTensor hidden, AccelTensor lmHeadW, int seqLen) {
        AccelTensor first = selectTokenAt(hidden, 0);
        AccelTensor last = selectTokenAt(hidden, seqLen - 1);
        AccelTensor firstLogits = null;
        AccelTensor lastLogits = null;
        try {
            firstLogits = linear(first, lmHeadW, null);
            lastLogits = linear(last, lmHeadW, null);
            int firstTop = topIndex(firstLogits.toFloatArray());
            int lastTop = topIndex(lastLogits.toFloatArray());
            System.err.printf("[DEBUG] Positional logits: firstTokenTop=%d lastTokenTop=%d%n", firstTop, lastTop);
            System.err.flush();
        } finally {
            if (firstLogits != null) {
                firstLogits.close();
            }
            if (lastLogits != null) {
                lastLogits.close();
            }
            first.close();
            last.close();
        }
    }

    private int topIndex(float[] logits) {
        int best = -1;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    // ── Linear ────────────────────────────────────────────────────────

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        return linear(input, weight, bias, null);
    }

    private AccelTensor ffnDownLinear(AccelTensor input,
                                      AccelTensor weight,
                                      AccelTensor bias,
                                      ModelConfig config,
                                      String profileKey) {
        return ffnDownLinear(input, weight, bias, config, profileKey, null);
    }

    private AccelTensor ffnDownLinear(AccelTensor input,
                                      AccelTensor weight,
                                      AccelTensor bias,
                                      ModelConfig config,
                                      String profileKey,
                                      AccelTensor outputBuffer) {
        if (canUseExperimentalMetalLinear()) {
            return linear(input, weight, bias, profileKey, config, outputBuffer);
        }
        AccelTensor cached = tryCachedFfnDownHalfLinear(input, weight, bias, config, profileKey);
        if (cached != null) {
            return cached;
        }
        return linear(input, weight, bias, profileKey, config, outputBuffer);
    }

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey) {
        return linear(input, weight, bias, profileKey, null);
    }

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
                               ModelConfig config) {
        return linear(input, weight, bias, profileKey, config, null);
    }

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
                               ModelConfig config, AccelTensor outputBuffer) {
        long t0 = profileKey != null ? System.nanoTime() : 0L;
        if (weight.isQuantized()
                && weight.quantType() != AccelTensor.QuantType.BF16
                && weight.quantType() != AccelTensor.QuantType.F16) {
            AccelTensor dequantized = weight.dequantize();
            try {
                return linear(input, dequantized, bias, profileKey, config, outputBuffer);
            } finally {
                if (dequantized != weight && !dequantized.isClosed()) {
                    dequantized.close();
                }
            }
        }
        AccelTensor metalHalf = tryMetalHalfLinear(input, weight, bias, config, outputBuffer, profileKey);
        if (metalHalf != null) {
            if (profileKey != null) {
                DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalHalf;
        }
        AccelTensor metalFloat = tryMetalFloatLinear(input, weight, bias);
        if (metalFloat != null) {
            if (profileKey != null) {
                DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return metalFloat;
        }
        AccelTensor cachedLargeHalf = tryCachedLargeHalfLinear(input, weight, bias, profileKey);
        if (cachedLargeHalf != null) {
            if (profileKey != null) {
                DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return cachedLargeHalf;
        }
        AccelTensor mm = AccelOps.linear(input, weight);
        if (bias != null) {
            AccelTensor out = AccelOps.add(mm, bias);
            mm.close();
            if (profileKey != null) {
                DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            }
            return out;
        }
        if (profileKey != null) {
            DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
        }
        return mm;
    }

    private AccelTensor tryCachedFfnDownHalfLinear(AccelTensor input,
                                                   AccelTensor weight,
                                                   AccelTensor bias,
                                                   ModelConfig config,
                                                   String profileKey) {
        if (!"ffn_down".equals(profileKey) && !"ffn_down_nongated".equals(profileKey)) {
            return null;
        }
        if (!isSingleTokenHalfLinearCandidate(input, weight)) {
            return null;
        }
        long perTensorMaxBytes = FFN_DOWN_LARGE_HALF_CACHE_PER_TENSOR_MAX_BYTES;
        if (perTensorMaxBytes <= 0L || weight.dequantizedByteSize() > perTensorMaxBytes) {
            return null;
        }
        long totalMaxBytes = FFN_DOWN_LARGE_HALF_CACHE_TOTAL_MAX_BYTES;
        if (totalMaxBytes <= 0L) {
            return null;
        }
        long estimatedModelBytes = multiplySaturating(weight.dequantizedByteSize(), config.numHiddenLayers());
        if (estimatedModelBytes > totalMaxBytes) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(perTensorMaxBytes);
        if (dequantized == weight) {
            return null;
        }
        return linear(input, dequantized, bias, profileKey);
    }

    private AccelTensor tryCachedLargeHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey) {
        if (!"logits".equals(profileKey)) {
            return null;
        }
        if (!isSingleTokenHalfLinearCandidate(input, weight)) {
            return null;
        }
        long maxBytes = LOGITS_LARGE_HALF_CACHE_MAX_BYTES;
        if (maxBytes <= 0L || weight.dequantizedByteSize() > maxBytes) {
            return null;
        }
        AccelTensor dequantized = weight.dequantizeCachedUpTo(maxBytes);
        if (dequantized == weight) {
            return null;
        }
        return linear(input, dequantized, bias, profileKey);
    }

    private boolean isSingleTokenHalfLinearCandidate(AccelTensor input, AccelTensor weight) {
        if (weight.quantType() != AccelTensor.QuantType.BF16
                && weight.quantType() != AccelTensor.QuantType.F16) {
            return false;
        }
        long[] inputShape = input.shape();
        if (inputShape.length < 1) {
            return false;
        }
        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        return rows == 1L;
    }

    private long multiplySaturating(long left, long right) {
        try {
            return Math.multiplyExact(left, right);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }

    private boolean canUseExperimentalMetalLinear() {
        return canUseMetal() && EXPERIMENTAL_METAL_LINEAR_ENABLED;
    }

    private static boolean resolveExperimentalMetalLinearEnabled() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // Default ON for Apple Silicon safetensor text path to avoid accidental CPU-only runs.
        return true;
    }

    private static boolean allowMetalBf16Linear() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return false;
    }

    private boolean allowMetalBf16Linear(ModelConfig config) {
        if (isGemma4Text(config)) {
            if (Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY)) {
                return false;
            }
            String explicit = System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
            if (explicit != null && !explicit.isBlank()) {
                return Boolean.parseBoolean(explicit);
            }
            return true;
        }
        return allowMetalBf16Linear();
    }

    private static boolean preferNativeMetalBf16Linear() {
        return Boolean.getBoolean(PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY);
    }

    private boolean preferNativeMetalBf16Linear(ModelConfig config) {
        if (isGemma4Text(config)) {
            if (allowGemma4Bf16ToF16Linear(config)) {
                return false;
            }
            String explicit = System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
            if (explicit != null && !explicit.isBlank()) {
                return Boolean.parseBoolean(explicit) && allowMetalBf16Linear(config);
            }
            return allowMetalBf16Linear(config);
        }
        return preferNativeMetalBf16Linear();
    }

    private boolean shouldUseNativeMetalBf16Linear(AccelTensor weight, ModelConfig config) {
        return preferNativeMetalBf16Linear(config)
                && allowMetalBf16Linear(config)
                && weight != null
                && weight.quantType() == AccelTensor.QuantType.BF16;
    }

    private boolean canUseMetalHalfLinearCandidate(AccelTensor input, AccelTensor weight, ModelConfig config) {
        if (!canUseExperimentalMetalLinear()) {
            return false;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (weight == null) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && isGemma4Text(config)
                && !shouldUseNativeMetalBf16Linear(weight, config)
                && !allowGemma4Bf16ToF16Linear(config)) {
            return false;
        }
        if (quantType != AccelTensor.QuantType.F16
                && (quantType != AccelTensor.QuantType.BF16 || !allowMetalBf16Linear(config))) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        long[] inputShape = input.shape();
        if (inputShape.length < 2 || weight.shape().length != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, inputShape[inputShape.length - 1]);
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        return batchProduct == 1L;
    }

    private boolean allowGemma4Bf16ToF16Linear(ModelConfig config) {
        if (!isGemma4Text(config) || Boolean.getBoolean(DISABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_GEMMA4_BF16_TO_F16_LINEAR_PROPERTY);
        return explicit != null && !explicit.isBlank() && Boolean.parseBoolean(explicit);
    }

    private boolean canUseMetalFloatLinearCandidate(AccelTensor input, AccelTensor weight) {
        if (!canUseExperimentalMetalLinear()) {
            return false;
        }
        if (input == null || weight == null) {
            return false;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        long[] inputShape = input.shape();
        if (inputShape.length < 2 || weight.shape().length != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, inputShape[inputShape.length - 1]);
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        return batchProduct == 1L;
    }

    private AccelTensor tryMetalHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
                                           ModelConfig config) {
        return tryMetalHalfLinear(input, weight, bias, config, null, null);
    }

    private AccelTensor tryMetalHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
                                           ModelConfig config, AccelTensor outputBuffer) {
        return tryMetalHalfLinear(input, weight, bias, config, outputBuffer, null);
    }

    private AccelTensor tryMetalHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
                                           ModelConfig config, AccelTensor outputBuffer, String profileKey) {
        if (!canUseMetalHalfLinearCandidate(input, weight, config)) {
            return null;
        }
        boolean nativeBf16Weight = shouldUseNativeMetalBf16Linear(weight, config);
        AccelTensor metalWeight = toMetalHalfWeight(weight, nativeBf16Weight, config);
        if (metalWeight == null) {
            return null;
        }
        long[] inputShape = input.shape();
        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = metalWeight.shape()[0];
        AccelTensor out = reusableOutputTensor(outputBuffer, outputShape);

        try {
            long k = inputShape[inputShape.length - 1];
            long rows = input.numel() / Math.max(1L, k);
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(metalWeight.shape()[0]);
            int rc = -2;
            if (m == 1
                    && nativeBf16Weight
                    && shouldUseMetalHalfMatvec(config, n, profileKey)
                    && metalBinding.supportsMatvecTransposedRightBf16()) {
                rc = metalBinding.matvecTransposedRightBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && shouldUseMetalLogitsMpsMatvec(config, n, kk, profileKey)
                    && metalBinding.supportsMatvecTransposedRightHalfMps()) {
                rc = metalBinding.matvecTransposedRightHalfMps(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && shouldUseMetalTransposedHalfMatvec(config, n, profileKey)
                    && metalBinding.supportsMatvecTransposedWeightHalf()) {
                AccelTensor transposedWeight = weight.toF16Transposed2dCachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
                if (transposedWeight != null
                        && transposedWeight.size(0) == k
                        && transposedWeight.size(1) == metalWeight.shape()[0]) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && shouldUseMetalHalfMatvec(config, n, profileKey)
                    && metalBinding.supportsMatvecTransposedRightHalf()) {
                rc = metalBinding.matvecTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalf(
                    out.dataPtr(),
                    contiguousInput.dataPtr(),
                    metalWeight.dataPtr(),
                    m, kk, n,
                    1.0f, 0.0f,
                    nativeBf16Weight);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalf failed with code " + rc);
            }
            if (bias == null) {
                return out;
            }
            AccelTensor biased = AccelOps.add(out, bias);
            out.close();
            return biased;
        } catch (RuntimeException e) {
            out.close();
            log.debugf("Falling back from Metal half linear to AccelOps: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor reusableWorkspaceView(java.lang.foreign.MemorySegment segment, long[] shape) {
        long requiredBytes = elementCount(shape) * Float.BYTES;
        if (segment == null || segment.byteSize() < requiredBytes) {
            return null;
        }
        return AccelTensor.view(segment, shape);
    }

    private AccelTensor reusableOutputTensor(AccelTensor outputBuffer, long[] outputShape) {
        if (outputBuffer != null
                && !outputBuffer.isClosed()
                && sameShape(outputBuffer.shape(), outputShape)) {
            return outputBuffer;
        }
        return AccelTensor.zeros(outputShape);
    }

    private static boolean canReuseFfnProjectionWorkspace() {
        if (Boolean.getBoolean(DISABLE_REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(REUSE_FFN_PROJECTION_WORKSPACE_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
    }

    private static boolean sameShape(long[] left, long[] right) {
        if (left.length != right.length) {
            return false;
        }
        for (int i = 0; i < left.length; i++) {
            if (left[i] != right[i]) {
                return false;
            }
        }
        return true;
    }

    private static long elementCount(long[] shape) {
        long count = 1L;
        for (long dim : shape) {
            count = Math.multiplyExact(count, dim);
        }
        return count;
    }

    private AccelTensor tryMetalFloatLinear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        if (!canUseMetalFloatLinearCandidate(input, weight)) {
            return null;
        }
        long[] inputShape = input.shape();
        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = weight.shape()[0];
        AccelTensor out = AccelTensor.zeros(outputShape);

        try {
            long k = inputShape[inputShape.length - 1];
            long rows = input.numel() / Math.max(1L, k);
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(weight.shape()[0]);
            int rc = metalBinding.matmulTransposedRight(
                    out.dataPtr(),
                    contiguousInput.dataPtr(),
                    weight.dataPtr(),
                    m, kk, n,
                    1.0f, 0.0f);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRight failed with code " + rc);
            }
            if (bias == null) {
                return out;
            }
            AccelTensor biased = AccelOps.add(out, bias);
            out.close();
            return biased;
        } catch (RuntimeException e) {
            out.close();
            log.debugf("Falling back from Metal float linear to AccelOps: %s", e.getMessage());
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private static AccelTensor resolveBias(Map<String, AccelTensor> weights, String key) {
        return key != null ? weights.get(key) : null;
    }

    private boolean canUseMetal() {
        if (Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY)) {
            return false;
        }
        return metalReady;
    }

    private boolean canUseMetalElementwise(ModelConfig config, int seqLen) {
        if (Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY)) {
            return false;
        }
        boolean gemma4 = isGemma4Text(config);
        if (gemma4) {
            if (Boolean.getBoolean(DISABLE_METAL_GEMMA4_ELEMENTWISE_PROPERTY)) {
                return false;
            }
            if (!canUseMetal()
                    || metalBinding == null
                    || !metalBinding.nativeElementwiseKernelsAvailable()) {
                return false;
            }
        } else if (!canUseMetal() && !canUseNativeElementwiseFallback()) {
            return false;
        }
        int defaultMinSeq = gemma4 ? 1 : 16;
        if (seqLen < Integer.getInteger(METAL_ELEMENTWISE_MIN_SEQ_PROPERTY, defaultMinSeq)) {
            return false;
        }
        if (!gemma4) {
            return true;
        }
        return Boolean.getBoolean(ENABLE_GEMMA4_METAL_ELEMENTWISE_PROPERTY)
                || metalBinding.nativeElementwiseKernelsAvailable();
    }

    private boolean canUseNativeElementwiseFallback() {
        return metalBinding != null && metalBinding.nativeElementwiseFallbackAvailable();
    }

    private boolean canUseNativeElementwiseAdd(ModelConfig config, int seqLen) {
        return canUseMetalElementwise(config, seqLen)
                && metalBinding.nativeElementwiseKernelsAvailable();
    }

    private boolean shouldUseMetalPostFfnNorm(ModelConfig config) {
        if (Boolean.getBoolean(DISABLE_METAL_POST_FFN_NORM_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_POST_FFN_NORM_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return isGemma4Text(config) || hasGemma4StylePerLayerInputs(config);
    }

    private boolean hasGemma4StylePerLayerInputs(ModelConfig config) {
        return config != null
                && (config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0);
    }

    private void rmsNormRowsMetal(java.lang.foreign.MemorySegment out, java.lang.foreign.MemorySegment in,
            java.lang.foreign.MemorySegment weight, int rows, int hiddenSize, float eps, boolean addOne) {
        int rc = metalBinding.rmsNormRows(out, in, weight, rows, hiddenSize, eps, addOne);
        if (rc != 0) {
            throw new IllegalStateException("Metal rmsNormRows failed with code " + rc);
        }
    }

    private void residualAdd(java.lang.foreign.MemorySegment left, AccelTensor right,
            java.lang.foreign.MemorySegment out, long[] shape, int seqLen, long hiddenSize, boolean useMetalElementwise) {
        if (useMetalElementwise) {
            try {
                int elements = Math.toIntExact(seqLen * hiddenSize);
                int rc = metalBinding.add(out, left, right.dataPtr(), elements);
                if (rc != 0) {
                    throw new IllegalStateException("Metal add failed with code " + rc);
                }
                return;
            } catch (IllegalStateException | UnsupportedOperationException e) {
                log.debugf("Falling back from Metal add to direct CPU accumulation: %s", e.getMessage());
            }
        }

        addSegments(left, right.dataPtr(), out, seqLen * hiddenSize);
    }

    private void addSegments(java.lang.foreign.MemorySegment left, java.lang.foreign.MemorySegment right,
            java.lang.foreign.MemorySegment out, long elements) {
        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(elements);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector leftVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, left, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector rightVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, right, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            leftVec.add(rightVec).intoMemorySegment(out, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < elements; i++) {
            float sum = left.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i)
                    + right.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, sum);
        }
    }

    private AccelTensor[] buildPerLayerInputs(long[] inputIds, AccelTensor inputsEmbeds,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch) {
        if (isGemma4Text(config) && Boolean.getBoolean(DISABLE_GEMMA4_PER_LAYER_INPUT_PROPERTY)) {
            return null;
        }
        if (config.hiddenSizePerLayerInput() <= 0) {
            return null;
        }

        AccelTensor packedPleEmbeddings = weights.get(arch.embedTokensPerLayerWeight());
        AccelTensor pleProjection = weights.get(arch.perLayerModelProjectionWeight());
        AccelTensor pleProjectionNorm = weights.get(arch.perLayerProjectionNormWeight());
        if (packedPleEmbeddings == null || pleProjection == null || pleProjectionNorm == null) {
            return null;
        }

        int numLayers = config.numHiddenLayers();
        int pleDim = config.hiddenSizePerLayerInput();
        int seqLen = inputIds.length;

        AccelTensor tokenPleRaw = packedPleEmbeddings.indexSelect(inputIds)
                .reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor tokenPle = AccelOps.mulScalar(tokenPleRaw, (float) Math.sqrt(Math.max(1, pleDim)));
        tokenPleRaw.close();
        AccelTensor projectedPle = linear(inputsEmbeds, pleProjection, null, "ple_projection", config);
        float pleProjectionScale = (float) (1.0 / Math.sqrt(Math.max(1, config.hiddenSize())));
        AccelTensor projectedPleScaled = AccelOps.mulScalar(projectedPle, pleProjectionScale);
        projectedPle.close();

        AccelTensor projectedPle4d = projectedPleScaled.reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor projectedPleNormed = AccelOps.rmsNorm(projectedPle4d, pleProjectionNorm,
                config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        projectedPle4d.close();

        AccelTensor combinedPle = AccelOps.add(projectedPleNormed, tokenPle);
        projectedPleNormed.close();
        tokenPle.close();

        AccelTensor scaledPle = AccelOps.mulScalar(combinedPle, (float) (1.0 / Math.sqrt(2.0)));
        combinedPle.close();

        AccelTensor[] layers = new AccelTensor[numLayers];
        if (seqLen == 1) {
            java.lang.foreign.MemorySegment packed = scaledPle.dataPtr();
            long layerBytes = (long) pleDim * Float.BYTES;
            for (int layer = 0; layer < numLayers; layer++) {
                layers[layer] = AccelTensor.view(
                        packed.asSlice((long) layer * layerBytes, layerBytes),
                        new long[] { 1L, 1L, pleDim },
                        scaledPle);
            }
            return layers;
        }

        java.lang.foreign.MemorySegment src = scaledPle.dataPtr();
        for (int layer = 0; layer < numLayers; layer++) {
            AccelTensor layerTensor = AccelTensor.zeros(new long[] { 1L, seqLen, pleDim });
            java.lang.foreign.MemorySegment dst = layerTensor.dataPtr();
            for (int token = 0; token < seqLen; token++) {
                long srcIndex = ((long) token * numLayers + layer) * pleDim;
                long dstIndex = (long) token * pleDim;
                java.lang.foreign.MemorySegment.copy(src, srcIndex * Float.BYTES, dst, dstIndex * Float.BYTES,
                        (long) pleDim * Float.BYTES);
            }
            layers[layer] = layerTensor;
        }
        scaledPle.close();
        return layers;
    }

    private void applyPerLayerResidual(java.lang.foreign.MemorySegment hiddenSeg, long[] hiddenShape, int seqLen,
            ModelConfig config, ModelArchitecture arch, Map<String, AccelTensor> weights, int layerIdx,
            AccelTensor perLayerInput) {
        if (perLayerInput == null) {
            return;
        }

        AccelTensor gateWeight = weights.get(arch.layerPerLayerInputGateWeight(layerIdx));
        AccelTensor projectionWeight = weights.get(arch.layerPerLayerProjectionWeight(layerIdx));
        AccelTensor normWeight = weights.get(arch.layerPostPerLayerInputNormWeight(layerIdx));
        if (gateWeight == null || projectionWeight == null || normWeight == null) {
            return;
        }

        AccelTensor hidden = AccelTensor.view(hiddenSeg, hiddenShape);
        AccelTensor gate = linear(hidden, gateWeight, null, "per_layer_gate", config);
        hidden.close();

        if (arch.activationType() == FFNActivationType.GELU) {
            AccelTensor activated = AccelOps.gelu(gate);
            gate.close();
            gate = activated;
        } else {
            AccelTensor activated = AccelOps.silu(gate);
            gate.close();
            gate = activated;
        }
        AccelTensor mixed = AccelOps.mul(gate, perLayerInput);
        gate.close();

        AccelTensor projected = linear(mixed, projectionWeight, null, "per_layer_projection", config);
        mixed.close();

        AccelTensor normed = AccelOps.rmsNorm(projected, normWeight,
                config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        projected.close();

        residualAdd(hiddenSeg, normed, hiddenSeg, hiddenShape, seqLen, config.hiddenSize(),
                canUseNativeElementwiseAdd(config, seqLen));
        normed.close();
    }

    private void closePerLayerInputs(AccelTensor[] perLayerInputs) {
        if (perLayerInputs == null) {
            return;
        }
        for (AccelTensor perLayerInput : perLayerInputs) {
            if (perLayerInput != null) {
                perLayerInput.closeWithParent();
            }
        }
    }

    private boolean useAddOneRmsNorm(ModelArchitecture arch, ModelConfig config) {
        if (config != null && config.modelType() != null) {
            String mt = config.modelType().toLowerCase();
            if (mt.startsWith("gemma3")) {
                // Gemma3 reference RMSNorm uses output * (1 + weight).
                return true;
            }
        }
        return arch.addOneToRmsNormWeight() && !isGemma4Text(config);
    }

    private float resolveEmbeddingScale(ModelArchitecture arch, ModelConfig config, int hiddenDim) {
        return arch.embeddingScaleFactor(hiddenDim);
    }

    private boolean shouldApplyLayerScalar(ModelConfig config) {
        if (!isGemma4Text(config)) {
            return true;
        }
        if (Boolean.getBoolean(DISABLE_GEMMA4_LAYER_SCALAR_PROPERTY)) {
            return false;
        }
        return true;
    }

    private boolean isGemma4Text(ModelConfig config) {
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        return modelType.startsWith("gemma4");
    }

    private void scaleSegmentInPlace(java.lang.foreign.MemorySegment seg, long[] shape, int seqLen, long hiddenSize, float scale) {
        long elements = seqLen * hiddenSize;
        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(elements);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector vec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, seg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            vec.mul(scale).intoMemorySegment(seg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < elements; i++) {
            float value = seg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            seg.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, value * scale);
        }
    }

    private float readScalarValue(AccelTensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor is null");
        }
        if (tensor.quantType() == AccelTensor.QuantType.F32) {
            return tensor.dataPtr().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        }
        AccelTensor dequantized = tensor.dequantize();
        try {
            return dequantized.dataPtr().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        } finally {
            if (dequantized != tensor && !dequantized.isClosed()) {
                dequantized.close();
            }
        }
    }

    private boolean shouldUseMetalHalfMatvec(ModelConfig config, int outputDim) {
        return shouldUseMetalHalfMatvec(config, outputDim, null);
    }

    private boolean shouldUseMetalHalfMatvec(ModelConfig config, int outputDim, String profileKey) {
        if (DISABLE_METAL_HALF_MATVEC_ENABLED) {
            return false;
        }
        int maxOutput = metalHalfMatvecMaxOutput(config, profileKey);
        String explicit = ENABLE_METAL_HALF_MATVEC_VALUE;
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit) && maxOutput > 0 && outputDim <= maxOutput;
        }
        return shouldAutoUseMetalHalfMatvec(config, outputDim, maxOutput);
    }

    private boolean shouldUseMetalHalfMatvecPair(ModelConfig config, int outputDim) {
        if (DISABLE_METAL_HALF_MATVEC_PAIR_ENABLED) {
            return false;
        }
        String explicit = ENABLE_METAL_HALF_MATVEC_PAIR_VALUE;
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit) && outputDim <= metalHalfMatvecMaxOutput();
        }
        return shouldAutoUseMetalHalfMatvec(config, outputDim);
    }

    private boolean shouldUseMetalLogitsMpsMatvec(ModelConfig config, int outputDim, int inputDim, String profileKey) {
        if (!"logits".equals(profileKey) || Boolean.getBoolean(DISABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_LOGITS_MPS_MATVEC_PROPERTY);
        if (explicit == null || explicit.isBlank() || !Boolean.parseBoolean(explicit)) {
            return false;
        }
        if (isGemma4Text(config)) {
            return false;
        }
        int minOutput = Integer.getInteger(
                METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT_PROPERTY,
                DEFAULT_METAL_LOGITS_MPS_MATVEC_MIN_OUTPUT);
        int maxInput = Integer.getInteger(
                METAL_LOGITS_MPS_MATVEC_MAX_INPUT_PROPERTY,
                DEFAULT_METAL_LOGITS_MPS_MATVEC_MAX_INPUT);
        return outputDim >= minOutput && (maxInput <= 0 || inputDim <= maxInput);
    }

    private boolean shouldAutoUseMetalHalfMatvec(ModelConfig config, int outputDim) {
        return shouldAutoUseMetalHalfMatvec(config, outputDim, metalHalfMatvecMaxOutput());
    }

    private boolean shouldAutoUseMetalHalfMatvec(ModelConfig config, int outputDim, int maxOutput) {
        return AUTO_METAL_HALF_MATVEC_ENABLED
                && maxOutput > 0
                && outputDim <= maxOutput
                && isMetalHalfMatvecAutoCandidate(config);
    }

    private boolean isMetalHalfMatvecAutoCandidate(ModelConfig config) {
        if (config == null || config.modelType() == null || config.modelType().isBlank()) {
            return false;
        }
        String modelType = config.modelType().toLowerCase();
        if (isGemma4Text(config) || hasGemma4StylePerLayerInputs(config)) {
            return true;
        }
        if (modelType.contains("qwen")) {
            return config.numHiddenLayers() >= 20
                    && config.hiddenSize() <= 2048
                    && config.intermediateSize() >= 2048;
        }
        return config.numHiddenLayers() >= 30
                && config.intermediateSize() >= 4096
                && config.hiddenSize() <= 4096;
    }

    private static int metalHalfMatvecMaxOutput() {
        return Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY, DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT);
    }

    private int metalHalfMatvecMaxOutput(ModelConfig config, String profileKey) {
        if ("logits".equals(profileKey) && isGemma4Text(config)) {
            return Integer.getInteger(
                    GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                    DEFAULT_GEMMA4_LOGITS_METAL_HALF_MATVEC_MAX_OUTPUT);
        }
        return metalHalfMatvecMaxOutput();
    }

    private boolean shouldUseMetalTransposedHalfMatvec(ModelConfig config, int outputDim, String profileKey) {
        if (Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY)) {
            return false;
        }
        int maxOutput = Integer.getInteger(
                METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT);
        if (maxOutput <= 0 || outputDim > maxOutput) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return "logits".equals(profileKey) && isGemma4Text(config);
    }

}
