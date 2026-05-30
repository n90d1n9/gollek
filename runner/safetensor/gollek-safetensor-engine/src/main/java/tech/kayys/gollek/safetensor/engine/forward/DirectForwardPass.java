/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.metal.binding.MetalBinding;

import java.util.Map;

/**
 * Full transformer forward pass using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectForwardPass {
    private static final Logger log = Logger.getLogger(DirectForwardPass.class);
    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;
    
    private MetalBinding metalBinding;
    private boolean metalReady;
    private DirectForwardMetalCapabilities metalCapabilities = DirectForwardMetalCapabilities.EMPTY;
    private final DirectForwardWeightResolver weightResolver = new DirectForwardWeightResolver();
    private volatile ModelConfigTraits lastModelConfigTraits = ModelConfigTraits.EMPTY;

    @jakarta.annotation.PostConstruct
    void init() {
        try {
            MetalBinding.initialize();
            this.metalBinding = MetalBinding.getInstance();
            this.metalBinding.init();
            String deviceName = this.metalBinding.deviceName();
            this.metalReady = this.metalBinding.isRuntimeActive()
                    && deviceName != null
                    && !deviceName.contains("CPU");
            refreshMetalCapabilities();
        } catch (Exception e) {
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.metalReady = false;
            refreshMetalCapabilities();
        }
    }

    private void refreshMetalCapabilities() {
        this.metalCapabilities = DirectForwardMetalCapabilities.detect(this.metalBinding);
    }

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(inputIds, weights, config, arch, kvCache);
        float[] result = DirectForwardLogits.materializeAndClose(logits);
        DirectForwardLogits.logPrefillDiagnostics(
                result,
                config,
                DirectForwardExecutionOptions.verboseTokensEnabled());
        return result;
    }

    public void clearResolvedModelWeights(Map<String, AccelTensor> weights) {
        weightResolver.clear(weights);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return prefill(embeddings, inputIds, null, weights, config, arch, kvCache);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache);
        return DirectForwardLogits.materializeAndClose(logits);
    }

    public AccelTensor prefillLogitsTensor(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        ResolvedModelWeights resolvedWeights = resolveModelWeights(weights, config, arch);
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.prefillTokenIds(
                runtime,
                attentionKernel,
                moeForwardPass,
                modelConfigTraits(config, arch),
                inputIds,
                weights,
                config,
                arch,
                kvCache,
                resolvedWeights,
                operators(runtime));
    }

    public AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        return prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache,
                resolveModelWeights(weights, config, arch));
    }

    private AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache, ResolvedModelWeights resolvedWeights) {
        return prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache,
                resolvedWeights, false);
    }

    private AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache, ResolvedModelWeights resolvedWeights,
            boolean embeddingsAlreadyInWorkspace) {
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.prefillEmbeddings(
                runtime,
                attentionKernel,
                moeForwardPass,
                modelConfigTraits(config, arch),
                embeddings,
                inputIds,
                perLayerInputs,
                weights,
                config,
                arch,
                kvCache,
                resolvedWeights,
                embeddingsAlreadyInWorkspace,
                operators(runtime));
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache);
        return DirectForwardLogits.materializeAndClose(logits);
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache, false);
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache,
            boolean reuseLogitsOutput) {
        ResolvedModelWeights resolvedWeights = resolveModelWeights(weights, config, arch);
        DirectForwardRuntimeContext runtime = runtimeContext();
        return DirectForwardSequenceRunner.decodeToken(
                runtime,
                attentionKernel,
                moeForwardPass,
                modelConfigTraits(config, arch),
                tokenId,
                startPos,
                weights,
                config,
                arch,
                kvCache,
                resolvedWeights,
                reuseLogitsOutput,
                operators(runtime));
    }

    AccelTensor ffnNonGated(AccelTensor x, ModelConfig config, AccelTensor upW, AccelTensor downW) {
        DirectForwardOperators operators = operators(runtimeContext());
        AccelTensor up = operators.linear(x, upW, null, "ffn_up_nongated", config);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = operators.ffnDownLinear(act, downW, null, config, "ffn_down_nongated");
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
        return operators(runtimeContext()).swigluFfn(x, arch, config, gateW, gateB, upW, upB, downW, downB, ws,
                downOutputBuffer);
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        return DirectForwardTensorOps.embeddingLookup(embedTable, tokenIds);
    }

    private DirectForwardRuntimeContext runtimeContext() {
        boolean metalLinearEnabled = !DirectForwardExecutionOptions.forceCpuForwardEnabled()
                && metalReady
                && DirectForwardMetalLinearPolicy.experimentalMetalLinearEnabled();
        return new DirectForwardRuntimeContext(
                log,
                metalBinding,
                metalCapabilities,
                metalReady,
                metalLinearEnabled);
    }

    private DirectForwardOperators operators(DirectForwardRuntimeContext runtime) {
        return new DirectForwardOperators(runtime, this::modelConfigTraits);
    }

    private ResolvedModelWeights resolveModelWeights(Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch) {
        boolean addOneRmsNorm = useAddOneRmsNorm(arch, config);
        return weightResolver.resolve(weights, config, arch, addOneRmsNorm);
    }

    private boolean useAddOneRmsNorm(ModelArchitecture arch, ModelConfig config) {
        ModelConfigTraits traits = modelConfigTraits(config, arch);
        if (traits.gemma3Text()) {
            // Gemma3 reference RMSNorm uses output * (1 + weight).
            return true;
        }
        return arch.addOneToRmsNormWeight() && !traits.gemma4Text();
    }

    private ModelConfigTraits modelConfigTraits(ModelConfig config) {
        return modelConfigTraits(config, null);
    }

    private ModelConfigTraits modelConfigTraits(ModelConfig config, ModelArchitecture arch) {
        if (config == null) {
            return ModelConfigTraits.EMPTY;
        }
        ModelConfigTraits cached = lastModelConfigTraits;
        if (cached.matches(config)) {
            return cached;
        }
        ModelConfigTraits traits = ModelConfigTraits.create(config, arch);
        lastModelConfigTraits = traits;
        return traits;
    }

}
