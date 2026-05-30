package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;
import org.jboss.logging.Logger;

@ApplicationScoped
public class FlashAttentionKernel {
    private static final Logger LOG = Logger.getLogger(FlashAttentionKernel.class);

    private MetalFlashAttentionBinding metalFa4;
    private MetalBinding metalBinding;
    private boolean metalReady;
    private FlashAttentionProjector projector;
    private FlashAttentionRoutingPolicy routingPolicy;
    private FlashAttentionNormalizer normalizer;
    private FlashAttentionRopeStage ropeStage;
    private FlashAttentionMetalAttention metalAttention;

    @Inject
    RopeFrequencyCache ropeCache;

    @jakarta.annotation.PostConstruct
    void init() {
        // Ensure bindings are initialized before first access; when dylib is absent,
        // both bindings should transparently enter CPU fallback mode.
        try {
            MetalBinding.initialize();
            this.metalBinding = MetalBinding.getInstance();
            this.metalBinding.init();
            String deviceName = this.metalBinding.deviceName();
            this.metalReady = this.metalBinding.isRuntimeActive()
                    && deviceName != null
                    && !deviceName.contains("CPU");
        } catch (Exception e) {
            LOG.warnf("FlashAttentionKernel: failed to initialize MetalBinding, forcing CPU fallback (%s)", e.getMessage());
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.metalReady = false;
        }
        this.projector = new FlashAttentionProjector(this.metalBinding, this::canUseMetal);
        this.routingPolicy = new FlashAttentionRoutingPolicy(this::canUseMetal, () -> this.metalBinding,
                () -> this.metalFa4);
        this.normalizer = new FlashAttentionNormalizer(() -> this.metalBinding);
        this.ropeStage = new FlashAttentionRopeStage(this.ropeCache);
        try {
            MetalFlashAttentionBinding.initialize();
            this.metalFa4 = MetalFlashAttentionBinding.getInstance();
        } catch (Exception e) {
            LOG.warnf("FlashAttentionKernel: failed to initialize MetalFlashAttentionBinding, forcing CPU fallback (%s)", e.getMessage());
            MetalFlashAttentionBinding.initializeFallback();
            this.metalFa4 = MetalFlashAttentionBinding.getInstance();
        }
        this.metalAttention = createMetalAttention();
    }

    public AccelTensor compute(AttentionInput in) {
        ModelConfig config = in.config;
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(in.arch, config);
        int layerIdx = in.layerIdx;
        int startPos = in.startPos;
        int seqLen = (int) in.x.size(1);
        FlashAttentionHeadLayout headLayout = FlashAttentionHeadLayout.resolve(in, config, layerIdx);
        int numQHeads = headLayout.numQueryHeads();
        int headDim = headLayout.headDim();
        int numKVHeads = headLayout.numKeyValueHeads();
        boolean gemma4Text = modelPolicy.gemma4Text();
        boolean alternativeAttention = config.usesAlternativeAttentionForLayer(layerIdx);
        boolean packedQkv = in.arch != null && in.arch.hasFusedQKV();
        // Gemma3 uses query_pre_attn_scalar^-0.5. For classic models the
        // parsed fallback is head_dim, preserving the usual head_dim^-0.5.
        float scale = gemma4Text ? 1.0f : (float) (1.0 / Math.sqrt(config.queryPreAttnScalar()));
        float attnSoftCap = modelPolicy.resolveAttentionSoftCap();
        boolean sharedKv = config.usesSharedKvCache(layerIdx);
        int kvLayerIdx = config.sharedKvSourceLayer(layerIdx);
        SharedKvState sharedKvState = sharedKv && in.sharedKvStates != null ? in.sharedKvStates.get(kvLayerIdx) : null;
        boolean useDenseSharedKvState = sharedKvState != null;
        boolean storeSharedKvState = !sharedKv
                && in.sharedKvStates != null
                && config.isSharedKvSourceLayer(layerIdx);
        FlashAttentionProjector attentionProjector = projector();
        FlashAttentionProjector.ProjectionBuffers qkvBuffers = attentionProjector.attentionProjectionBuffers(in,
                !packedQkv && !sharedKv && !useDenseSharedKvState && !alternativeAttention);

        // 1. Projections
        FlashAttentionProjector.LinearTriple qkvTriple = packedQkv && !sharedKv && !useDenseSharedKvState
                && !alternativeAttention
                ? attentionProjector.projectPackedQkv(in, config, modelPolicy, headLayout)
                : null;
        if (qkvTriple == null
                && !packedQkv
                && !sharedKv
                && !useDenseSharedKvState
                && !alternativeAttention) {
            qkvTriple = attentionProjector.tryMetalHalfLinearTripleMixed(
                    in.x, in.qW, in.qB, in.kW, in.kB, in.vW, in.vB, "attn_qkv_proj_triple", config,
                    modelPolicy,
                    qkvBuffers == null ? null : qkvBuffers.q(),
                    qkvBuffers == null ? null : qkvBuffers.k(),
                    qkvBuffers == null ? null : qkvBuffers.v());
        }
        FlashAttentionProjector.LinearPair qkPair = qkvTriple == null && !packedQkv
                && (!sharedKv && !useDenseSharedKvState)
                ? attentionProjector.tryMetalHalfLinearPairMixed(
                        in.x, in.qW, in.qB, in.kW, in.kB, "attn_qk_proj_pair", config,
                        modelPolicy)
                : null;
        AccelTensor q = qkvTriple != null ? qkvTriple.first()
                : (qkPair != null ? qkPair.first() : attentionProjector.project(
                        in.x, in.qW, in.qB, "attn_q_proj", config,
                        modelPolicy));
        AccelTensor k = useDenseSharedKvState
                ? sharedKvState.key()
                : (sharedKv ? null : (qkvTriple != null ? qkvTriple.second()
                        : (qkPair != null ? qkPair.second() : attentionProjector.project(
                                in.x, in.kW, in.kB, "attn_k_proj", config,
                                modelPolicy))));
        AccelTensor v = useDenseSharedKvState
                ? sharedKvState.value()
                : (sharedKv
                        ? null
                        : (alternativeAttention ? AccelTensor.copyOf(k.dataPtr(), k.shape())
                                : (qkvTriple != null ? qkvTriple.third()
                                        : attentionProjector.project(
                                                in.x, in.vW, in.vB, "attn_v_proj", config, modelPolicy))));

        // 3. Reshape and RoPE
        AccelTensor q4 = q.reshape(in.x.size(0), in.x.size(1), numQHeads, headDim);
        q.close();
        q = q4;
        if (!sharedKv) {
            AccelTensor k4 = k.reshape(in.x.size(0), in.x.size(1), numKVHeads, headDim);
            k.close();
            k = k4;
        }

        // QK-Norm (Per-head)
        FlashAttentionNormalizer attentionNormalizer = normalizer();
        boolean addOneRmsNorm = in.arch.addOneToRmsNormWeight() && !gemma4Text;
        boolean disableGemma4QkNorm = attentionNormalizer.gemma4QkNormDisabled(modelPolicy);
        if (!disableGemma4QkNorm && in.qNormW != null) {
            AccelTensor qNormed = attentionNormalizer.perHeadRmsNorm(q, in.qNormW, config.rmsNormEps(), addOneRmsNorm,
                    modelPolicy);
            q.close();
            q = qNormed;
        }
        if (!disableGemma4QkNorm && !sharedKv && in.kNormW != null) {
            AccelTensor kNormed = attentionNormalizer.perHeadRmsNorm(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm,
                    modelPolicy);
            k.close();
            k = kNormed;
        }

        if (!sharedKv) {
            AccelTensor v4 = v.reshape(in.x.size(0), in.x.size(1), numKVHeads, headDim);
            v.close();
            v = v4;
            if (gemma4Text && !attentionNormalizer.gemma4VNormDisabled(modelPolicy)) {
                AccelTensor vNormed = attentionNormalizer.perHeadRmsNormNoWeight(v, config.rmsNormEps(), modelPolicy);
                v.close();
                v = vNormed;
            }
        }

        ropeStage().apply(q, sharedKv ? null : k, startPos, config, modelPolicy, layerIdx, headDim);

        // 4. Update Cache
        KVCacheManager.KVCacheSession kvSession = in.kvCache;
        if (!sharedKv) {
            PagedKvCacheIO.updateCache(k, v, kvSession, layerIdx, startPos, seqLen, numKVHeads, headDim);
        }
        if (storeSharedKvState) {
            SharedKvState appended = appendSharedKvState(in.sharedKvStates.get(layerIdx), k, v);
            in.sharedKvStates.put(layerIdx, appended);
            k = appended.key();
            v = appended.value();
        }

        // 5. Attention
        FlashAttentionRoutingPolicy routing = routingPolicy();
        AccelTensor attnOut;
        if (useDenseSharedKvState) {
            AccelTensor metalSharedOut = metalAttention().denseSharedAttention(q, k, v, sharedKvState, config,
                    modelPolicy,
                    layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, in.isCausal, attnSoftCap);
            if (metalSharedOut != null) {
                attnOut = metalSharedOut;
            } else {
                DirectInferenceProfiler.recordAttentionPath("dense_shared_java");
                attnOut = FlashAttentionJavaFallback.denseSharedAttention(q, k, v, config, layerIdx, startPos,
                        numQHeads, numKVHeads, headDim, scale, in.isCausal, attnSoftCap);
            }
        } else if (routing.canUseDenseGemma4Attention(config, modelPolicy, layerIdx) && !kvSession.isQuantized()) {
            DirectInferenceProfiler.recordAttentionPath("dense_gemma4_java");
            attnOut = FlashAttentionJavaFallback.denseCachedAttention(q, kvSession, kvLayerIdx, startPos, numQHeads,
                    numKVHeads, headDim, scale, in.isCausal, attnSoftCap, config, layerIdx);
        } else if (routing.canUseFa4PagedAttention(config, layerIdx, attnSoftCap)) {
            attnOut = metalAttention().tiledAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads,
                    headDim, scale, in.isCausal, attnSoftCap, config, modelPolicy, layerIdx);
        } else if (routing.canUseSlidingDecodeMetalAttention(config, modelPolicy, layerIdx, seqLen)) {
            attnOut = metalAttention().slidingDecodeAttention(q, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads,
                    numKVHeads, headDim, scale, attnSoftCap, config, modelPolicy);
        } else if (routing.canUseMetalAttention(config, modelPolicy, layerIdx, seqLen, startPos, attnSoftCap)) {
            attnOut = metalAttention().tiledAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads,
                    headDim, scale, in.isCausal, attnSoftCap, config, modelPolicy, layerIdx);
        } else {
            DirectInferenceProfiler.recordAttentionPath("paged_java");
            attnOut = PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads,
                    numKVHeads, headDim, scale, in.isCausal, attnSoftCap);
        }
        q.close();
        if (useDenseSharedKvState && sharedKvState != null) {
            sharedKvState.releaseView(k);
            sharedKvState.releaseView(v);
        } else {
            if (k != null && !storeSharedKvState) k.close();
            if (v != null && !storeSharedKvState) v.close();
        }

        // Reshape back to 3D for subsequent operations
        AccelTensor attnOut3 = attnOut.reshape(in.x.size(0), seqLen, numQHeads * headDim);
        attnOut.close();
        attnOut = attnOut3;

        // 6. Output Projection: maps [B,T,numQHeads*headDim] → [B,T,hidden_size]
        AccelTensor projectionBuffer = attentionProjector.attentionOutputBufferView(in, attnOut);
        AccelTensor projected = attentionProjector.project(attnOut, in.oW, in.oB, "attn_o_proj", config, modelPolicy,
                projectionBuffer);
        if (projectionBuffer != null && projectionBuffer != projected && !projectionBuffer.isClosed()) {
            projectionBuffer.close();
        }
        attnOut.close();

        // Post-Attention Norm (Gemma-2/4) applies to the attention output
        // before the residual add in the decoder layer.
        if (in.postAttnNormW != null) {
            AccelTensor normed = attentionNormalizer.rmsNorm(projected, in.postAttnNormW, config.rmsNormEps(),
                    addOneRmsNorm);
            projected.close();
            projected = normed;
        }

        return projected;
    }

    private SharedKvState appendSharedKvState(SharedKvState existing, AccelTensor deltaKey, AccelTensor deltaValue) {
        if (existing == null) {
            return new SharedKvState(deltaKey, deltaValue);
        }
        existing.append(deltaKey, deltaValue);
        return existing;
    }

    private boolean canUseMetal() {
        if (FlashAttentionRuntimeOptions.forceCpuForwardEnabled()) {
            return false;
        }
        return metalReady;
    }

    private FlashAttentionProjector projector() {
        if (projector == null) {
            projector = new FlashAttentionProjector(metalBinding, this::canUseMetal);
        }
        return projector;
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        if (routingPolicy == null) {
            routingPolicy = new FlashAttentionRoutingPolicy(this::canUseMetal, () -> metalBinding,
                    () -> metalFa4);
        }
        return routingPolicy;
    }

    private FlashAttentionNormalizer normalizer() {
        if (normalizer == null) {
            normalizer = new FlashAttentionNormalizer(() -> metalBinding);
        }
        return normalizer;
    }

    private FlashAttentionRopeStage ropeStage() {
        if (ropeStage == null) {
            ropeStage = new FlashAttentionRopeStage(ropeCache);
        }
        return ropeStage;
    }

    private FlashAttentionMetalAttention metalAttention() {
        if (metalAttention == null) {
            metalAttention = createMetalAttention();
        }
        return metalAttention;
    }

    private FlashAttentionMetalAttention createMetalAttention() {
        return new FlashAttentionMetalAttention(this::canUseMetal, () -> metalBinding, () -> metalFa4,
                this::routingPolicy);
    }

}
