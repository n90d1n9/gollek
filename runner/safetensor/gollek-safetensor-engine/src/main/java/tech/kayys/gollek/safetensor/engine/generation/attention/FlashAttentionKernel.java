package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import org.jboss.logging.Logger;

@ApplicationScoped
public class FlashAttentionKernel {
    private static final Logger LOG = Logger.getLogger(FlashAttentionKernel.class);
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final String ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_gemma4_metal_bf16_linear";
    private static final String DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_gemma4_metal_bf16_linear";
    private static final String METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY =
            "gollek.safetensor.metal_f16_weight_cache_max_bytes";
    private static final long DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    private static final long METAL_F16_WEIGHT_CACHE_MAX_BYTES = Long.getLong(
            METAL_F16_WEIGHT_CACHE_MAX_BYTES_PROPERTY,
            DEFAULT_METAL_F16_WEIGHT_CACHE_MAX_BYTES);
    private static final boolean EXPERIMENTAL_METAL_LINEAR_ENABLED =
            resolveExperimentalMetalLinearEnabled();
    private static final String ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.allow_metal_gemma4_attention";
    private static final String DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_attention";
    private static final String ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY =
            "gollek.safetensor.allow_legacy_metal_attention_bridge";
    private static final String ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_shared_decode_packed_attention";
    private static final String DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_shared_decode_packed_attention";
    private static final String ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_paged_decode_attention";
    private static final String DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_paged_decode_attention";
    private static final String ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_raw_paged_sliding_decode_attention";
    private static final String ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_sliding_prefill_fa4_attention";
    private static final String DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_gemma4_sliding_prefill_fa4_attention";
    private static final String PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_attention_max_tokens";
    private static final int DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS = 1024;
    private static final String DISABLE_GEMMA4_V_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_v_norm";
    private static final String DISABLE_GEMMA4_QK_NORM_PROPERTY =
            "gollek.safetensor.disable_gemma4_qk_norm";
    private static final String ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.enable_metal_per_head_rms_norm";
    private static final String DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY =
            "gollek.safetensor.disable_metal_per_head_rms_norm";
    private static final String EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY =
            "gollek.safetensor.experimental_gemma4_split_half_rope";
    private static final String LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY =
            "gollek.safetensor.legacy_interleaved_gemma4_rope";
    private static final String FORCE_DENSE_GEMMA4_ATTENTION_PROPERTY =
            "gollek.safetensor.force_dense_gemma4_attention";
    private static final String ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_pair";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple";
    private static final String DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_mixed_half_linear_triple_matvec";
    private static final String METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_mixed_half_linear_triple_matvec_max_output";
    private static final int DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT = 4096;
    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final String AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_attention_half_matvec";
    private static final boolean AUTO_METAL_HALF_MATVEC_ENABLED =
            resolveAutoMetalHalfMatvecEnabled();
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 8192;
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 262144;
    private static final boolean METAL_MIXED_HALF_LINEAR_PAIR_ENABLED =
            resolveMetalMixedHalfLinearPairEnabled();
    private static final int MAX_PACKED_METAL_SLIDING_WINDOW = 2048;

    private MetalFlashAttentionBinding metalFa4;
    private MetalBinding metalBinding;
    private boolean metalReady;
    
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
            this.metalReady = this.metalBinding.isNativeAvailable()
                    && deviceName != null
                    && !deviceName.contains("CPU");
        } catch (Exception e) {
            LOG.warnf("FlashAttentionKernel: failed to initialize MetalBinding, forcing CPU fallback (%s)", e.getMessage());
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.metalReady = false;
        }
        try {
            MetalFlashAttentionBinding.initialize();
            this.metalFa4 = MetalFlashAttentionBinding.getInstance();
        } catch (Exception e) {
            LOG.warnf("FlashAttentionKernel: failed to initialize MetalFlashAttentionBinding, forcing CPU fallback (%s)", e.getMessage());
            MetalFlashAttentionBinding.initializeFallback();
            this.metalFa4 = MetalFlashAttentionBinding.getInstance();
        }
    }

    public static class AttentionInput {
        public final AccelTensor x;
        public final AccelTensor qW, kW, vW, oW;
        public final AccelTensor qB, kB, vB, oB;
        public final ModelArchitecture arch;
        public final ModelConfig config;
        public final KVCacheManager.KVCacheSession kvCache;
        public final int layerIdx;
        public final int startPos;
        public final boolean isCausal;
        public final AccelTensor qNormW, kNormW;
        public final AccelTensor postAttnNormW;
        public final Map<Integer, SharedKvState> sharedKvStates;

        public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
                AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
                ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
                int layerIdx, int startPos, boolean isCausal,
                AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW,
                Map<Integer, SharedKvState> sharedKvStates) {
            this.x = x;
            this.qW = qW;
            this.kW = kW;
            this.vW = vW;
            this.oW = oW;
            this.qB = qB;
            this.kB = kB;
            this.vB = vB;
            this.oB = oB;
            this.arch = arch;
            this.config = config;
            this.kvCache = kvCache;
            this.layerIdx = layerIdx;
            this.startPos = startPos;
            this.isCausal = isCausal;
            this.qNormW = qNormW;
            this.kNormW = kNormW;
            this.postAttnNormW = postAttnNormW;
            this.sharedKvStates = sharedKvStates;
        }
    }

    public static class SharedKvState {
        private AccelTensor keyBuffer;
        private AccelTensor valueBuffer;
        private AccelTensor packedKeyBuffer;
        private AccelTensor packedValueBuffer;
        private final int numHeads;
        private final int headDim;
        private int lengthTokens;
        private int capacityTokens;

        public SharedKvState(AccelTensor key, AccelTensor value) {
            this(key, value, Math.toIntExact(key.size(2)), Math.toIntExact(key.size(3)),
                    Math.toIntExact(key.size(1)), Math.toIntExact(key.size(1)));
        }

        private SharedKvState(AccelTensor keyBuffer, AccelTensor valueBuffer,
                int numHeads, int headDim, int lengthTokens, int capacityTokens) {
            this.keyBuffer = keyBuffer;
            this.valueBuffer = valueBuffer;
            this.numHeads = numHeads;
            this.headDim = headDim;
            this.lengthTokens = lengthTokens;
            this.capacityTokens = capacityTokens;
        }

        public AccelTensor key() {
            return viewTokens(keyBuffer, lengthTokens);
        }

        public AccelTensor value() {
            return viewTokens(valueBuffer, lengthTokens);
        }

        public void releaseView(AccelTensor tensor) {
            if (tensor != null
                    && tensor != keyBuffer
                    && tensor != valueBuffer
                    && !tensor.isClosed()) {
                tensor.close();
            }
        }

        public void append(AccelTensor deltaKey, AccelTensor deltaValue) {
            int deltaTokens = Math.toIntExact(deltaKey.size(1));
            if (deltaTokens <= 0) {
                closeIfView(deltaKey, keyBuffer);
                closeIfView(deltaValue, valueBuffer);
                return;
            }

            ensureCapacity(lengthTokens + deltaTokens);
            appendIntoBuffer(keyBuffer, deltaKey, lengthTokens, numHeads, headDim);
            appendIntoBuffer(valueBuffer, deltaValue, lengthTokens, numHeads, headDim);
            if (packedKeyBuffer != null && packedValueBuffer != null) {
                appendIntoPackedBuffer(packedKeyBuffer, deltaKey, lengthTokens, numHeads, headDim);
                appendIntoPackedBuffer(packedValueBuffer, deltaValue, lengthTokens, numHeads, headDim);
            }
            lengthTokens += deltaTokens;
            closeIfView(deltaKey, keyBuffer);
            closeIfView(deltaValue, valueBuffer);
        }

        public MemorySegment packedKeyData() {
            ensurePackedBuffers();
            return packedKeyBuffer.dataPtr();
        }

        public MemorySegment packedValueData() {
            ensurePackedBuffers();
            return packedValueBuffer.dataPtr();
        }

        public int packedCapacityTokens() {
            ensurePackedBuffers();
            return capacityTokens;
        }

        private void ensureCapacity(int requiredTokens) {
            if (requiredTokens <= capacityTokens) {
                return;
            }
            int newCapacity = Math.max(requiredTokens, Math.max(4, capacityTokens * 2));
            keyBuffer = growBuffer(keyBuffer, newCapacity, numHeads, headDim, lengthTokens);
            valueBuffer = growBuffer(valueBuffer, newCapacity, numHeads, headDim, lengthTokens);
            if (packedKeyBuffer != null) {
                packedKeyBuffer = growPackedBuffer(packedKeyBuffer, newCapacity, numHeads, headDim, lengthTokens);
            }
            if (packedValueBuffer != null) {
                packedValueBuffer = growPackedBuffer(packedValueBuffer, newCapacity, numHeads, headDim, lengthTokens);
            }
            capacityTokens = newCapacity;
        }

        private void ensurePackedBuffers() {
            if (packedKeyBuffer != null && packedValueBuffer != null) {
                return;
            }
            long batch = keyBuffer.size(0);
            packedKeyBuffer = AccelTensor.zeros(batch, numHeads, capacityTokens, headDim);
            packedValueBuffer = AccelTensor.zeros(batch, numHeads, capacityTokens, headDim);
            packBuffer(keyBuffer, packedKeyBuffer, lengthTokens, numHeads, headDim);
            packBuffer(valueBuffer, packedValueBuffer, lengthTokens, numHeads, headDim);
        }

        private static AccelTensor growBuffer(AccelTensor existing, int newCapacity, int numHeads, int headDim,
                int currentLength) {
            long batch = existing.size(0);
            AccelTensor grown = AccelTensor.zeros(batch, newCapacity, numHeads, headDim);
            copyTokenRange(existing, grown, currentLength, numHeads, headDim);
            existing.close();
            return grown;
        }

        private static AccelTensor growPackedBuffer(AccelTensor existing, int newCapacity, int numHeads, int headDim,
                int currentLength) {
            long batch = existing.size(0);
            AccelTensor grown = AccelTensor.zeros(batch, numHeads, newCapacity, headDim);
            copyPackedTokenRange(existing, grown, currentLength, numHeads, headDim);
            existing.close();
            return grown;
        }

        private static void appendIntoBuffer(AccelTensor destination, AccelTensor delta, int tokenOffset,
                int numHeads, int headDim) {
            long batch = destination.size(0);
            int deltaTokens = Math.toIntExact(delta.size(1));
            long perTokenBytes = (long) numHeads * headDim * Float.BYTES;
            long srcBatchStrideBytes = (long) deltaTokens * perTokenBytes;
            long dstBatchStrideBytes = destination.size(1) * perTokenBytes;
            MemorySegment src = delta.dataPtr();
            MemorySegment dst = destination.dataPtr();
            for (int b = 0; b < batch; b++) {
                long srcOffset = b * srcBatchStrideBytes;
                long dstOffset = b * dstBatchStrideBytes + (long) tokenOffset * perTokenBytes;
                MemorySegment.copy(src, srcOffset, dst, dstOffset, srcBatchStrideBytes);
            }
        }

        private static void appendIntoPackedBuffer(AccelTensor destination, AccelTensor delta, int tokenOffset,
                int numHeads, int headDim) {
            long batch = destination.size(0);
            int deltaTokens = Math.toIntExact(delta.size(1));
            MemorySegment src = delta.dataPtr();
            MemorySegment dst = destination.dataPtr();
            long deltaBatchStrideBytes = (long) deltaTokens * numHeads * headDim * Float.BYTES;
            long deltaTokenStrideBytes = (long) numHeads * headDim * Float.BYTES;
            long deltaHeadStrideBytes = (long) headDim * Float.BYTES;
            long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
            long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
            long copyBytes = (long) headDim * Float.BYTES;
            for (int b = 0; b < batch; b++) {
                long srcBatchBase = b * deltaBatchStrideBytes;
                long dstBatchBase = b * dstBatchStrideBytes;
                for (int tok = 0; tok < deltaTokens; tok++) {
                    long srcTokenBase = srcBatchBase + tok * deltaTokenStrideBytes;
                    long dstTokenBase = dstBatchBase + (long) (tokenOffset + tok) * headDim * Float.BYTES;
                    for (int h = 0; h < numHeads; h++) {
                        long srcOffset = srcTokenBase + h * deltaHeadStrideBytes;
                        long dstOffset = dstTokenBase + h * dstHeadStrideBytes;
                        MemorySegment.copy(src, srcOffset, dst, dstOffset, copyBytes);
                    }
                }
            }
        }

        private static void copyTokenRange(AccelTensor source, AccelTensor destination, int tokenCount,
                int numHeads, int headDim) {
            if (tokenCount <= 0) {
                return;
            }
            long batch = source.size(0);
            long copyBytesPerBatch = (long) tokenCount * numHeads * headDim * Float.BYTES;
            long srcBatchStrideBytes = source.size(1) * (long) numHeads * headDim * Float.BYTES;
            long dstBatchStrideBytes = destination.size(1) * (long) numHeads * headDim * Float.BYTES;
            MemorySegment src = source.dataPtr();
            MemorySegment dst = destination.dataPtr();
            for (int b = 0; b < batch; b++) {
                MemorySegment.copy(src, b * srcBatchStrideBytes, dst, b * dstBatchStrideBytes, copyBytesPerBatch);
            }
        }

        private static void copyPackedTokenRange(AccelTensor source, AccelTensor destination, int tokenCount,
                int numHeads, int headDim) {
            if (tokenCount <= 0) {
                return;
            }
            long batch = source.size(0);
            long copyBytesPerHead = (long) tokenCount * headDim * Float.BYTES;
            long srcBatchStrideBytes = (long) numHeads * source.size(2) * headDim * Float.BYTES;
            long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
            long srcHeadStrideBytes = source.size(2) * (long) headDim * Float.BYTES;
            long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
            MemorySegment src = source.dataPtr();
            MemorySegment dst = destination.dataPtr();
            for (int b = 0; b < batch; b++) {
                long srcBatchBase = b * srcBatchStrideBytes;
                long dstBatchBase = b * dstBatchStrideBytes;
                for (int h = 0; h < numHeads; h++) {
                    MemorySegment.copy(src, srcBatchBase + h * srcHeadStrideBytes,
                            dst, dstBatchBase + h * dstHeadStrideBytes, copyBytesPerHead);
                }
            }
        }

        private static void packBuffer(AccelTensor source, AccelTensor destination, int tokenCount,
                int numHeads, int headDim) {
            if (tokenCount <= 0) {
                return;
            }
            long batch = source.size(0);
            MemorySegment src = source.dataPtr();
            MemorySegment dst = destination.dataPtr();
            long srcBatchStrideBytes = source.size(1) * (long) numHeads * headDim * Float.BYTES;
            long srcTokenStrideBytes = (long) numHeads * headDim * Float.BYTES;
            long srcHeadStrideBytes = (long) headDim * Float.BYTES;
            long dstBatchStrideBytes = (long) numHeads * destination.size(2) * headDim * Float.BYTES;
            long dstHeadStrideBytes = destination.size(2) * (long) headDim * Float.BYTES;
            long copyBytes = (long) headDim * Float.BYTES;
            for (int b = 0; b < batch; b++) {
                long srcBatchBase = b * srcBatchStrideBytes;
                long dstBatchBase = b * dstBatchStrideBytes;
                for (int tok = 0; tok < tokenCount; tok++) {
                    long srcTokenBase = srcBatchBase + tok * srcTokenStrideBytes;
                    long dstTokenBase = dstBatchBase + (long) tok * headDim * Float.BYTES;
                    for (int h = 0; h < numHeads; h++) {
                        long srcOffset = srcTokenBase + h * srcHeadStrideBytes;
                        long dstOffset = dstTokenBase + h * dstHeadStrideBytes;
                        MemorySegment.copy(src, srcOffset, dst, dstOffset, copyBytes);
                    }
                }
            }
        }

        private static AccelTensor viewTokens(AccelTensor buffer, int tokenCount) {
            if (tokenCount == Math.toIntExact(buffer.size(1))) {
                return buffer;
            }
            return buffer.slice(1, 0, tokenCount);
        }

        private static void closeIfView(AccelTensor tensor, AccelTensor backingBuffer) {
            if (tensor != null && tensor != backingBuffer && !tensor.isClosed()) {
                tensor.close();
            }
        }

        public void close() {
            if (keyBuffer != null && !keyBuffer.isClosed()) {
                keyBuffer.close();
            }
            if (valueBuffer != null && !valueBuffer.isClosed()) {
                valueBuffer.close();
            }
            if (packedKeyBuffer != null && !packedKeyBuffer.isClosed()) {
                packedKeyBuffer.close();
            }
            if (packedValueBuffer != null && !packedValueBuffer.isClosed()) {
                packedValueBuffer.close();
            }
        }
    }

    public AccelTensor compute(AttentionInput in) {
        ModelConfig config = in.config;
        int layerIdx = in.layerIdx;
        int startPos = in.startPos;
        int seqLen = (int) in.x.size(1);
        int numQHeads = config.numAttentionHeads();
        int headDim = (int) in.qW.size(0) / numQHeads;
        int numKVHeads = config.resolvedNumKvHeadsForLayer(layerIdx);
        boolean gemma4Text = isGemma4Text(config);
        boolean alternativeAttention = config.usesAlternativeAttentionForLayer(layerIdx);
        // Gemma3 uses query_pre_attn_scalar^-0.5. For classic models the
        // parsed fallback is head_dim, preserving the usual head_dim^-0.5.
        float scale = gemma4Text ? 1.0f : (float) (1.0 / Math.sqrt(config.queryPreAttnScalar()));
        float attnSoftCap = resolveAttentionSoftCap(in.arch, config, gemma4Text);
        boolean sharedKv = config.usesSharedKvCache(layerIdx);
        int kvLayerIdx = config.sharedKvSourceLayer(layerIdx);
        SharedKvState sharedKvState = sharedKv && in.sharedKvStates != null ? in.sharedKvStates.get(kvLayerIdx) : null;
        boolean useDenseSharedKvState = sharedKvState != null;
        boolean storeSharedKvState = !sharedKv
                && in.sharedKvStates != null
                && config.isSharedKvSourceLayer(layerIdx);

        // 1. Projections
        LinearTriple qkvTriple = (!sharedKv && !useDenseSharedKvState && !alternativeAttention)
                ? tryMetalHalfLinearTripleMixed(
                        in.x, in.qW, in.qB, in.kW, in.kB, in.vW, in.vB, "attn_qkv_proj_triple", config)
                : null;
        LinearPair qkPair = qkvTriple == null && (!sharedKv && !useDenseSharedKvState)
                ? tryMetalHalfLinearPairMixed(in.x, in.qW, in.qB, in.kW, in.kB, "attn_qk_proj_pair", config)
                : null;
        AccelTensor q = qkvTriple != null ? qkvTriple.first()
                : (qkPair != null ? qkPair.first() : project(in.x, in.qW, in.qB, "attn_q_proj", config));
        AccelTensor k = useDenseSharedKvState
                ? sharedKvState.key()
                : (sharedKv ? null : (qkvTriple != null ? qkvTriple.second()
                        : (qkPair != null ? qkPair.second() : project(in.x, in.kW, in.kB, "attn_k_proj", config))));
        AccelTensor v = useDenseSharedKvState
                ? sharedKvState.value()
                : (sharedKv
                        ? null
                        : (alternativeAttention ? AccelTensor.copyOf(k.dataPtr(), k.shape())
                                : (qkvTriple != null ? qkvTriple.third()
                                        : project(in.x, in.vW, in.vB, "attn_v_proj", config))));

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
        boolean addOneRmsNorm = in.arch.addOneToRmsNormWeight() && !gemma4Text;
        boolean disableGemma4QkNorm = gemma4Text && Boolean.getBoolean(DISABLE_GEMMA4_QK_NORM_PROPERTY);
        if (!disableGemma4QkNorm && in.qNormW != null) {
            AccelTensor qNormed = tryMetalPerHeadRmsNorm(q, in.qNormW, config.rmsNormEps(), addOneRmsNorm, config);
            if (qNormed == null) {
                qNormed = AccelOps.perHeadRmsNorm(q, in.qNormW, config.rmsNormEps(), addOneRmsNorm);
            }
            q.close();
            q = qNormed;
        }
        if (!disableGemma4QkNorm && !sharedKv && in.kNormW != null) {
            AccelTensor kNormed = tryMetalPerHeadRmsNorm(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm, config);
            if (kNormed == null) {
                kNormed = AccelOps.perHeadRmsNorm(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm);
            }
            k.close();
            k = kNormed;
        }

        if (!sharedKv) {
            AccelTensor v4 = v.reshape(in.x.size(0), in.x.size(1), numKVHeads, headDim);
            v.close();
            v = v4;
            if (gemma4Text && !Boolean.getBoolean(DISABLE_GEMMA4_V_NORM_PROPERTY)) {
                AccelTensor vNormed = perHeadRmsNormNoWeight(v, config.rmsNormEps());
                v.close();
                v = vNormed;
            }
        }

        int rotaryDim = resolveRotaryStorageDim(config, layerIdx, headDim);
        int rotatedDim = resolveRotatedDim(config, layerIdx, headDim, rotaryDim);
        RopeFrequencyCache.RopeFrequencies freqs = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(),
                config.ropeThetaForLayer(layerIdx), config.ropeScaling(),
                resolveRopeExponentDenominator(config, layerIdx, headDim, rotaryDim), rotatedDim);
        // Hugging Face Gemma-4 text uses split-half `rotate_half`, not the
        // legacy adjacent-pair interleaved rotation. Keep the old path only
        // as an explicit escape hatch while we finish the remaining parity
        // work around shared-KV and full/sliding attention interaction.
        boolean gemma4LegacyInterleavedRope = gemma4Text
                && Boolean.getBoolean(LEGACY_INTERLEAVED_GEMMA4_ROPE_PROPERTY);
        boolean gemma4SplitHalfRope = gemma4Text
                && (Boolean.getBoolean(EXPERIMENTAL_GEMMA4_SPLIT_HALF_ROPE_PROPERTY)
                        || !gemma4LegacyInterleavedRope);
        boolean gemma3Text = isGemma3Text(config);
        boolean interleavedRope = gemma4Text
                ? !gemma4SplitHalfRope
                : gemma3Text
                        ? false // Gemma3 aligns with HF rotate_half (split-half)
                        : !in.arch.usesNeoxRope();
        applyRope(q, sharedKv ? null : k, startPos, freqs, interleavedRope);

        // 4. Update Cache
        KVCacheManager.KVCacheSession kvSession = in.kvCache;
        if (!sharedKv) {
            updateKVCache(k, v, kvSession, layerIdx, startPos, seqLen, numKVHeads, headDim);
        }
        if (storeSharedKvState) {
            SharedKvState appended = appendSharedKvState(in.sharedKvStates.get(layerIdx), k, v);
            in.sharedKvStates.put(layerIdx, appended);
            k = appended.key();
            v = appended.value();
        }

        // 5. Attention
        AccelTensor attnOut;
        if (useDenseSharedKvState) {
            AccelTensor metalSharedOut = denseSharedMetalAttention(q, k, v, sharedKvState, config, layerIdx, startPos, numQHeads,
                    numKVHeads, headDim, scale, in.isCausal, attnSoftCap);
            DirectInferenceEngine.recordAttentionPath(
                    metalSharedOut != null ? "dense_shared_metal" : "dense_shared_java");
            attnOut = metalSharedOut != null
                    ? metalSharedOut
                    : denseSharedAttention(q, k, v, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale,
                            in.isCausal, attnSoftCap);
        } else if (canUseDenseGemma4Attention(config, layerIdx) && !kvSession.isQuantized()) {
            DirectInferenceEngine.recordAttentionPath("dense_gemma4_java");
            attnOut = denseCachedAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim, scale,
                    in.isCausal, attnSoftCap, config, layerIdx);
        } else if (canUseFa4PagedAttention(config, layerIdx, kvSession, attnSoftCap)) {
            attnOut = tiledAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim, scale,
                    in.isCausal, attnSoftCap, config, layerIdx);
        } else if (canUseSlidingDecodeMetalAttention(config, layerIdx, kvSession, seqLen)) {
            attnOut = slidingDecodeMetalAttention(q, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads,
                    numKVHeads, headDim, scale, attnSoftCap, config);
        } else if (canUseMetalAttention(config, layerIdx, kvSession, seqLen, startPos, attnSoftCap)) {
            attnOut = tiledAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim, scale,
                    in.isCausal, attnSoftCap, config, layerIdx);
        } else {
            DirectInferenceEngine.recordAttentionPath("paged_java");
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
        AccelTensor projected = project(attnOut, in.oW, in.oB, "attn_o_proj", config);
        attnOut.close();

        // Post-Attention Norm (Gemma-2/4) applies to the attention output
        // before the residual add in the decoder layer.
        if (in.postAttnNormW != null) {
            AccelTensor normed = rmsNorm(projected, in.postAttnNormW, config.rmsNormEps(), addOneRmsNorm);
            projected.close();
            projected = normed;
        }

        return projected;
    }

    private AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = Math.toIntExact(k.size(1));
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        int slidingWindow = slidingLayer ? config.slidingWindowSize() : Integer.MAX_VALUE;

        AccelTensor out = AccelTensor.zeros(q.shape());
        MemorySegment qSeg = q.dataSegment();
        MemorySegment kSeg = k.dataSegment();
        MemorySegment vSeg = v.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        long qStride0 = q.stride()[0];
        long qStride1 = q.stride()[1];
        long qStride2 = q.stride()[2];
        long kStride0 = k.stride()[0];
        long kStride1 = k.stride()[1];
        long kStride2 = k.stride()[2];
        long vStride0 = v.stride()[0];
        long vStride1 = v.stride()[1];
        long vStride2 = v.stride()[2];
        long oStride0 = out.stride()[0];
        long oStride1 = out.stride()[1];
        long oStride2 = out.stride()[2];

        int gqaGroup = numQHeads / Math.max(1, numKVHeads);

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numQHeads; h++) {
                int kvHeadIdx = h / gqaGroup;
                for (int i = 0; i < seqLenQ; i++) {
                    long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2);
                    float m = Float.NEGATIVE_INFINITY;
                    float l = 0.0f;
                    float[] acc = new float[headDim];
                    int minPos = slidingWindow == Integer.MAX_VALUE
                            ? 0
                            : Math.max(0, startPos + i - slidingWindow + 1);

                    for (int tok = 0; tok < totalTokens; tok++) {
                        if (tok < minPos) {
                            continue;
                        }
                        if (causal && tok > startPos + i) {
                            break;
                        }

                        long kOff = ((long) b * kStride0 + (long) tok * kStride1 + (long) kvHeadIdx * kStride2);
                        float score = dotProduct(qSeg, qOff * 4L, kSeg, kOff * 4L, headDim) * scale;
                        if (softCap > 0.0f) {
                            score = (float) (Math.tanh(score / softCap) * softCap);
                        }

                        float mPrev = m;
                        m = Math.max(m, score);
                        float expPrev = (float) Math.exp(mPrev - m);
                        float expCurr = (float) Math.exp(score - m);
                        l = l * expPrev + expCurr;

                        long vOff = ((long) b * vStride0 + (long) tok * vStride1 + (long) kvHeadIdx * vStride2);
                        updateAccumulator(acc, vSeg, vOff * 4L, expPrev, expCurr, headDim);
                    }

                    long oOff = ((long) b * oStride0 + (long) i * oStride1 + (long) h * oStride2);
                    for (int d = 0; d < headDim; d++) {
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, oOff + d, acc[d] / (l + 1e-9f));
                    }
                }
            }
        }

        return out;
    }

    private AccelTensor denseSharedMetalAttention(AccelTensor q, AccelTensor k, AccelTensor v, SharedKvState sharedKvState,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        if (!canUseMetal()) {
            return null;
        }
        if (metalBinding == null || !metalBinding.isNativeAvailable()) {
            return null;
        }
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = Math.toIntExact(k.size(1));
        if (batch <= 0 || seqLenQ <= 0 || totalTokens <= 0) {
            return null;
        }

        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        if (slidingLayer && !metalBinding.isWindowedAttentionAvailable()) {
            return null;
        }

        boolean usePackedSharedDecode = shouldUsePackedSharedDecodeAttention(config, seqLenQ, sharedKvState);
        boolean useFa4 = canUseFa4Attention(softCap) && !usePackedSharedDecode;
        if (!useFa4 && !usePackedSharedDecode && !allowLegacyMetalAttentionBridge(config)) {
            return null;
        }
        AccelTensor qContiguous = q.contiguous();
        AccelTensor kContiguous = null;
        AccelTensor vContiguous = null;
        try (Arena arena = Arena.ofConfined()) {
            AccelTensor out = AccelTensor.zeros(q.shape());

            if (useFa4) {
                kContiguous = k.contiguous();
                vContiguous = v.contiguous();
                boolean useBf16 = metalFa4.isBf16Available()
                        && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = metalFa4.fa4Attention(
                        out.dataPtr(), qContiguous.dataPtr(), kContiguous.dataPtr(), vContiguous.dataPtr(),
                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), totalTokens, numQHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    return out;
                }
                out.close();
            } else {
                int blockSize = Math.max(1, sharedKvState != null
                        ? sharedKvState.packedCapacityTokens()
                        : totalTokens);
                int maxBlocks = 1;
                MemorySegment packedK;
                MemorySegment packedV;
                if (sharedKvState != null) {
                    packedK = sharedKvState.packedKeyData();
                    packedV = sharedKvState.packedValueData();
                } else {
                    kContiguous = k.contiguous();
                    vContiguous = v.contiguous();
                    long blockElements = (long) maxBlocks * numKVHeads * blockSize * headDim;
                    packedK = arena.allocate(batch * blockElements * Float.BYTES, 64);
                    packedV = arena.allocate(batch * blockElements * Float.BYTES, 64);
                    packDenseSharedKvIntoTemporaryPagedPool(kContiguous, vContiguous, numKVHeads, headDim, packedK, packedV);
                }

                MemorySegment blockTable = arena.allocate(batch * maxBlocks * Integer.BYTES, Integer.BYTES);
                for (int b = 0; b < batch; b++) {
                    blockTable.setAtIndex(ValueLayout.JAVA_INT, b * maxBlocks, b);
                }

                MemorySegment contextLens = arena.allocate(batch * Integer.BYTES, Integer.BYTES);
                for (int b = 0; b < batch; b++) {
                    contextLens.setAtIndex(ValueLayout.JAVA_INT, b, totalTokens);
                }

                int result = slidingLayer
                        ? (numKVHeads == numQHeads
                                ? metalBinding.attentionWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap)
                                : metalBinding.attentionGqaWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap))
                        : (numKVHeads == numQHeads
                                ? metalBinding.attention(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap)
                                : metalBinding.attentionGqa(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap));
                if (result == 0) {
                    return out;
                }
                out.close();
            }
        } catch (RuntimeException e) {
            return null;
        } finally {
            if (qContiguous != null && qContiguous != q && !qContiguous.isClosed()) {
                qContiguous.close();
            }
            if (kContiguous != null && kContiguous != k && !kContiguous.isClosed()) {
                kContiguous.close();
            }
            if (vContiguous != null && vContiguous != v && !vContiguous.isClosed()) {
                vContiguous.close();
            }
        }
        return null;
    }

    private float dotProduct(MemorySegment q, long qOffBytes, MemorySegment k, long kOffBytes, int headDim) {
        long qIndex = qOffBytes / Float.BYTES;
        long kIndex = kOffBytes / Float.BYTES;
        int upperBound = FLOAT_SPECIES.loopBound(headDim);
        FloatVector sumVec = FloatVector.zero(FLOAT_SPECIES);
        int j = 0;
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector qVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, q, (qIndex + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector kVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, k, (kIndex + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            sumVec = qVec.fma(kVec, sumVec);
        }
        float res = sumVec.reduceLanes(VectorOperators.ADD);
        for (; j < headDim; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, qIndex + j)
                    * k.getAtIndex(ValueLayout.JAVA_FLOAT, kIndex + j);
        }
        return res;
    }

    private void updateAccumulator(float[] acc, MemorySegment vSeg, long vOffBytes, float expPrev, float expCurr,
            int headDim) {
        long vIndex = vOffBytes / Float.BYTES;
        int upperBound = FLOAT_SPECIES.loopBound(headDim);
        FloatVector prevVec = FloatVector.broadcast(FLOAT_SPECIES, expPrev);
        FloatVector currVec = FloatVector.broadcast(FLOAT_SPECIES, expCurr);
        int j = 0;
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector accVec = FloatVector.fromArray(FLOAT_SPECIES, acc, j);
            FloatVector valueVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, vSeg, (vIndex + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            accVec.mul(prevVec).add(valueVec.mul(currVec)).intoArray(acc, j);
        }
        for (; j < headDim; j++) {
            acc[j] = acc[j] * expPrev
                    + vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, vIndex + j) * expCurr;
        }
    }

    private AccelTensor rmsNorm(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        return AccelOps.rmsNorm(x, w, eps, addOne);
    }

    private AccelTensor tryMetalPerHeadRmsNorm(AccelTensor x, AccelTensor weight, double eps, boolean addOne,
            ModelConfig config) {
        if (!shouldUseMetalPerHeadRmsNorm(config) || x == null || weight == null) {
            return null;
        }
        if (metalBinding == null || !metalBinding.nativeElementwiseKernelsAvailable()) {
            return null;
        }
        if (x.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        long[] shape = x.shape();
        if (shape.length < 2) {
            return null;
        }
        int headDim = Math.toIntExact(shape[shape.length - 1]);
        if (headDim <= 0 || weight.numel() != headDim) {
            return null;
        }

        AccelTensor weightView = weight;
        AccelTensor contiguousInput = null;
        AccelTensor out = null;
        try {
            if (weight.quantType() != AccelTensor.QuantType.F32) {
                weightView = weight.dequantize();
            }
            if (weightView.quantType() != AccelTensor.QuantType.F32 || weightView.numel() != headDim) {
                return null;
            }
            contiguousInput = x.contiguous();
            out = AccelTensor.zeros(shape);
            int rows = Math.toIntExact(contiguousInput.numel() / headDim);
            int rc = metalBinding.rmsNormRows(out.dataPtr(), contiguousInput.dataPtr(), weightView.dataPtr(),
                    rows, headDim, (float) eps, addOne);
            if (rc == 0) {
                return out;
            }
            out.close();
            out = null;
            return null;
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (contiguousInput != null && contiguousInput != x && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
            if (weightView != weight && !weightView.isClosed()) {
                weightView.close();
            }
        }
    }

    private boolean shouldUseMetalPerHeadRmsNorm(ModelConfig config) {
        if (Boolean.getBoolean(DISABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_PER_HEAD_RMS_NORM_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // Gemma-4 performs Q/K per-head normalization on every layer. On M4 the
        // native rows kernel consistently wins despite the extra launch cost.
        return isGemma4Text(config);
    }

    private boolean shouldUsePackedSharedDecodeAttention(ModelConfig config, long seqLenQ,
            SharedKvState sharedKvState) {
        if (sharedKvState == null || seqLenQ != 1L) {
            return false;
        }
        if (!isGemma4Text(config)
                || Boolean.getBoolean(DISABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY)
                || Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_GEMMA4_SHARED_DECODE_PACKED_ATTENTION_PROPERTY);
        return explicit != null
                && !explicit.isBlank()
                && Boolean.parseBoolean(explicit)
                && metalBinding != null
                && metalBinding.isWindowedAttentionAvailable();
    }

    private SharedKvState appendSharedKvState(SharedKvState existing, AccelTensor deltaKey, AccelTensor deltaValue) {
        if (existing == null) {
            return new SharedKvState(deltaKey, deltaValue);
        }
        existing.append(deltaKey, deltaValue);
        return existing;
    }

    private void applyRope(AccelTensor q, AccelTensor k, int startPos, RopeFrequencyCache.RopeFrequencies freqs,
            boolean interleaved) {
        int seqLen = (int) q.size(1);
        int numQHeads = (int) q.size(2);
        int headDim = (int) q.size(3);

        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numQHeads; h++) {
                freqs.rotateInPlace(q.dataPtr(), ((long) s * numQHeads + h) * headDim, pos, interleaved);
            }
            if (k != null) {
                int numKVHeads = (int) k.size(2);
                for (int h = 0; h < numKVHeads; h++) {
                    freqs.rotateInPlace(k.dataPtr(), ((long) s * numKVHeads + h) * headDim, pos, interleaved);
                }
            }
        }
    }

    private int resolveRotaryStorageDim(ModelConfig config, int layerIdx, int headDim) {
        // Partial RoPE (Gemma-4 proportional full attention) still uses the
        // full head layout. Only the first rotated span receives non-identity
        // frequencies; the remaining split-half pairs must stay in place.
        return headDim;
    }

    private int resolveRotatedDim(ModelConfig config, int layerIdx, int headDim, int storageDim) {
        double partialFactor = config.partialRotaryFactorForLayer(layerIdx);
        int rotaryDim = (int) Math.round(storageDim * partialFactor);
        rotaryDim = Math.max(2, rotaryDim);
        if ((rotaryDim & 1) != 0) {
            rotaryDim--;
        }
        return Math.min(storageDim, rotaryDim);
    }

    private int resolveRopeExponentDenominator(ModelConfig config, int layerIdx, int headDim, int rotaryDim) {
        return rotaryDim;
    }

    private boolean isGemma4Text(ModelConfig config) {
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        return modelType.startsWith("gemma4");
    }

    private boolean hasGemma4StylePerLayerInputs(ModelConfig config) {
        return config != null
                && (config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0);
    }

    private boolean isGemma3Text(ModelConfig config) {
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        return modelType.startsWith("gemma3");
    }

    private AccelTensor perHeadRmsNormNoWeight(AccelTensor x, double eps) {
        x = x.contiguous();
        long[] shape = x.shape();
        int headDim = (int) shape[shape.length - 1];
        int numHeads = (int) shape[shape.length - 2];
        int outer = (int) (x.numel() / (numHeads * headDim));

        AccelTensor out = AccelTensor.zeros(shape);
        MemorySegment xSeg = x.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        for (int b = 0; b < outer; b++) {
            for (int h = 0; h < numHeads; h++) {
                long base = (long) (b * numHeads + h) * headDim;
                int upperBound = FLOAT_SPECIES.loopBound(headDim);
                FloatVector sumSqVec = FloatVector.zero(FLOAT_SPECIES);
                int i = 0;
                for (; i < upperBound; i += FLOAT_SPECIES.length()) {
                    FloatVector valueVec = FloatVector.fromMemorySegment(
                            FLOAT_SPECIES, xSeg, (base + i) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                    sumSqVec = valueVec.fma(valueVec, sumSqVec);
                }
                float sumSq = sumSqVec.reduceLanes(VectorOperators.ADD);
                for (; i < headDim; i++) {
                    float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                    sumSq += val * val;
                }
                float rms = (float) (1.0 / Math.sqrt(sumSq / headDim + eps));
                FloatVector rmsVec = FloatVector.broadcast(FLOAT_SPECIES, rms);
                int j = 0;
                for (; j < upperBound; j += FLOAT_SPECIES.length()) {
                    FloatVector valueVec = FloatVector.fromMemorySegment(
                            FLOAT_SPECIES, xSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                    valueVec.mul(rmsVec).intoMemorySegment(
                            oSeg, (base + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                }
                for (; j < headDim; j++) {
                    float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + j);
                    oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, base + j, val * rms);
                }
            }
        }
        return out;
    }

    private void updateKVCache(AccelTensor k, AccelTensor v, KVCacheManager.KVCacheSession kvSession, int layerIdx,
            int startPos, int seqLen, int numHeads, int headDim) {
        BlockManager blockManager = kvSession.blockManager();
        MemorySegment kSeg = k.dataPtr();
        MemorySegment vSeg = v.dataPtr();

        // Writes happen before kvCache.advance(...), so we must provision the
        // backing block tables for the whole write span up front.
        kvSession.ensureCapacity(startPos + seqLen);

        int tokensPerBlock = kvSession.tokensPerBlock();
        long headStride = blockManager.getHeadStride();
        long tokenStride = blockManager.getTokenStride();

        boolean quantizedInt8 = kvSession.isQuantizedInt8();
        boolean quantizedInt4 = kvSession.isQuantizedInt4();
        for (int s = 0; s < seqLen; s++) {
            int absolutePos = startPos + s;
            int blockIdx = kvSession.getBlockForToken(layerIdx, absolutePos);
            if (blockIdx < 0) {
                throw new IllegalStateException(
                        "Missing KV block for layer " + layerIdx + " token " + absolutePos
                                + " (startPos=" + startPos + ", seqLen=" + seqLen + ")");
            }
            int tokenIdxInBlock = absolutePos % tokensPerBlock;

            MemorySegment kBlock = blockManager.getKBlock(blockIdx);
            MemorySegment vBlock = blockManager.getVBlock(blockIdx);
            MemorySegment kScaleBlock = blockManager.getKScaleBlock(blockIdx);
            MemorySegment vScaleBlock = blockManager.getVScaleBlock(blockIdx);

            for (int h = 0; h < numHeads; h++) {
                long srcOff = ((long) s * numHeads + h) * headDim;
                long hDstOff = ((long) h * headStride + (long) tokenIdxInBlock * tokenStride);

                if (quantizedInt8) {
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokenIdxInBlock;
                    quantizeVectorToInt8(kSeg, srcOff, kBlock, hDstOff, kScaleBlock, scaleIndex, headDim);
                    quantizeVectorToInt8(vSeg, srcOff, vBlock, hDstOff, vScaleBlock, scaleIndex, headDim);
                } else if (quantizedInt4) {
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokenIdxInBlock;
                    quantizeVectorToInt4(kSeg, srcOff, kBlock, hDstOff, kScaleBlock, scaleIndex, headDim);
                    quantizeVectorToInt4(vSeg, srcOff, vBlock, hDstOff, vScaleBlock, scaleIndex, headDim);
                } else {
                    MemorySegment.copy(kSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, kBlock, ValueLayout.JAVA_FLOAT,
                            hDstOff * 4, headDim);
                    MemorySegment.copy(vSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, vBlock, ValueLayout.JAVA_FLOAT,
                            hDstOff * 4, headDim);
                }
            }
        }
    }

    private void quantizeVectorToInt8(MemorySegment srcSeg, long srcFloatIndex,
            MemorySegment dstSeg, long dstElementIndex,
            MemorySegment scaleSeg, long scaleIndex,
            int headDim) {
        float absmax = 0.0f;
        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            float abs = Math.abs(value);
            if (abs > absmax) {
                absmax = abs;
            }
        }

        float scale = absmax == 0.0f ? 1.0f : absmax / 127.0f;
        if (scaleSeg != null) {
            scaleSeg.setAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex, scale);
        }

        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            int quantized = Math.round(value / scale);
            quantized = Math.max(-127, Math.min(127, quantized));
            dstSeg.setAtIndex(ValueLayout.JAVA_BYTE, dstElementIndex + d, (byte) quantized);
        }
    }

    private void quantizeVectorToInt4(MemorySegment srcSeg, long srcFloatIndex,
            MemorySegment dstSeg, long dstElementIndex,
            MemorySegment scaleSeg, long scaleIndex,
            int headDim) {
        float absmax = 0.0f;
        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            float abs = Math.abs(value);
            if (abs > absmax) {
                absmax = abs;
            }
        }

        float scale = absmax == 0.0f ? 1.0f : absmax / 7.0f;
        if (scaleSeg != null) {
            scaleSeg.setAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex, scale);
        }

        for (int d = 0; d < headDim; d++) {
            float value = srcSeg.getAtIndex(ValueLayout.JAVA_FLOAT, srcFloatIndex + d);
            int quantized = Math.round(value / scale);
            quantized = Math.max(-8, Math.min(7, quantized));
            writePackedSignedInt4(dstSeg, dstElementIndex + d, quantized);
        }
    }

    private void writePackedSignedInt4(MemorySegment dstSeg, long dstElementIndex, int quantized) {
        long byteIndex = dstElementIndex >>> 1;
        int stored = quantized + 8;
        int existing = Byte.toUnsignedInt(dstSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int packed = (dstElementIndex & 1L) == 0L
                ? ((existing & 0xF0) | (stored & 0x0F))
                : ((existing & 0x0F) | ((stored & 0x0F) << 4));
        dstSeg.setAtIndex(ValueLayout.JAVA_BYTE, byteIndex, (byte) packed);
    }

    private AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        java.util.List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        boolean canUseFa4Path = canUseFa4Attention(softCap);
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        boolean preferPagedMetalFirst = preferPagedMetalAttentionBeforeFa4(config, (int) seqLen, totalTokens);

        if (preferPagedMetalFirst && !kvSession.isQuantized()) {
            AccelTensor pagedOut = tryPagedMetalAttention(q, kvSession, blockManager, blocks, kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens, batch,
                    seqLen, slidingLayer, null);
            if (pagedOut != null) {
                DirectInferenceEngine.recordAttentionPath(slidingLayer
                        ? "paged_metal_windowed_first"
                        : "paged_metal_first");
                return pagedOut;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            AccelTensor out = AccelTensor.zeros(q.shape());
            if (!preferPagedMetalFirst && canUseFa4Path && metalFa4 != null && metalFa4.isNativeAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                gatherKV(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim, kGathered, vGathered);

                boolean useBf16 = metalFa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = metalFa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceEngine.recordAttentionPath("fa4_gathered");
                    return out;
                }
                out.close();
                out = AccelTensor.zeros(q.shape());
            }
            AccelTensor pagedOut = tryPagedMetalAttention(q, kvSession, blockManager, blocks, kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens, batch,
                    seqLen, slidingLayer, arena);
            if (pagedOut != null) {
                out.close();
                DirectInferenceEngine.recordAttentionPath(slidingLayer ? "paged_metal_windowed" : "paged_metal");
                return pagedOut;
            }
            if (canUseFa4Path && metalFa4 != null && metalFa4.isNativeAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                gatherKV(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim, kGathered, vGathered);

                boolean useBf16 = metalFa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = metalFa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceEngine.recordAttentionPath("fa4_gathered_after_paged");
                    return out;
                }
                out.close();
            }
            DirectInferenceEngine.recordAttentionPath("paged_java");
            return PagedAttentionVectorAPI.compute(q, null, kvSession, kvLayerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, causal, softCap);
        }
    }

    private AccelTensor tryPagedMetalAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            BlockManager blockManager, java.util.List<Integer> blocks, int kvLayerIdx, int startPos, int numHeads,
            int numKVHeads, int headDim, float scale, boolean causal, float softCap, ModelConfig config, int layerIdx,
            int totalTokens, long batch, long seqLen, boolean slidingLayer, Arena arena) {
        if (!allowPagedMetalAttentionBridge(config, (int) seqLen, totalTokens)
                || metalBinding == null
                || !metalBinding.isNativeAvailable()
                || blocks == null
                || blocks.isEmpty()) {
            return null;
        }
        if (kvSession.isQuantized() && arena == null) {
            return null;
        }
        AccelTensor out = AccelTensor.zeros(q.shape());
        try {
            int maxBlocks = blocks.size();
            int batchInt = Math.toIntExact(batch);
            MemorySegment kPool = kvSession.getRawKPool();
            MemorySegment vPool = kvSession.getRawVPool();
            KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvSession.getWorkspace();
            ws.ensureAttentionMetadataCapacity(batchInt * maxBlocks, batchInt);
            MemorySegment blockTableSegment = ws.getAttentionBlockTableSeg();
            MemorySegment contextLensSegment = ws.getAttentionContextLensSeg();
            if (kvSession.isQuantized()) {
                MaterializedKvPools materialized = materializeKvBlocksForMetal(blockManager, kvSession, blocks,
                        numKVHeads, headDim, arena);
                kPool = materialized.kPool();
                vPool = materialized.vPool();
                for (int b = 0; b < batchInt; b++) {
                    int base = b * maxBlocks;
                    for (int i = 0; i < maxBlocks; i++) {
                        blockTableSegment.setAtIndex(ValueLayout.JAVA_INT, base + i, i);
                    }
                }
            } else {
                for (int b = 0; b < batchInt; b++) {
                    int base = b * maxBlocks;
                    for (int i = 0; i < maxBlocks; i++) {
                        blockTableSegment.setAtIndex(ValueLayout.JAVA_INT, base + i, blocks.get(i));
                    }
                }
            }
            for (int b = 0; b < batchInt; b++) {
                contextLensSegment.setAtIndex(ValueLayout.JAVA_INT, b, totalTokens);
            }

            int result = slidingLayer
                    ? (numKVHeads == numHeads
                            ? metalBinding.attentionWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap)
                            : metalBinding.attentionGqaWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap))
                    : (numKVHeads == numHeads
                            ? metalBinding.attention(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, softCap)
                            : metalBinding.attentionGqa(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, softCap));
            if (result == 0) {
                return out;
            }
        } catch (RuntimeException e) {
            // Fall through to FA4/Java attention fallback below.
        }
        out.close();
        return null;
    }

    private AccelTensor slidingDecodeMetalAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            int layerIdx, int kvLayerIdx, int startPos, int numHeads, int numKVHeads, int headDim, float scale,
            float softCap, ModelConfig config) {
        if (!allowSlidingDecodeMetalAttentionBridge(config, layerIdx, (int) q.size(1))) {
            DirectInferenceEngine.recordAttentionPath("sliding_decode_java");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
        long batch = q.size(0);
        int seqLen = (int) q.size(1);
        int totalTokens = startPos + seqLen;
        int slidingWindow = config.slidingWindowSize();
        int contextStart = Math.max(0, totalTokens - slidingWindow);
        int contextTokens = totalTokens - contextStart;
        int blockSize = kvSession.tokensPerBlock();
        int maxBlocks = Math.max(1, (contextTokens + blockSize - 1) / blockSize);
        int poolElements = maxBlocks * numKVHeads * blockSize * headDim;
        int batchInt = Math.toIntExact(batch);

        if (Boolean.getBoolean(ENABLE_RAW_PAGED_SLIDING_DECODE_ATTENTION_PROPERTY)) {
            AccelTensor rawPagedOut = tryPagedMetalAttention(q, kvSession, kvSession.blockManager(),
                    kvSession.getBlockIndices(kvLayerIdx), kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, true, softCap, config, layerIdx,
                    totalTokens, batch, seqLen, true, null);
            if (rawPagedOut != null) {
                DirectInferenceEngine.recordAttentionPath("sliding_decode_paged_metal");
                return rawPagedOut;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment packedK = arena.allocate((long) poolElements * Float.BYTES, 64);
            MemorySegment packedV = arena.allocate((long) poolElements * Float.BYTES, 64);
            packKvRangeIntoTemporaryPagedPool(kvSession.blockManager(), kvSession, kvLayerIdx, contextStart,
                    totalTokens, numKVHeads, headDim, blockSize, packedK, packedV);

            KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvSession.getWorkspace();
            ws.ensureAttentionMetadataCapacity(batchInt * maxBlocks, batchInt);
            MemorySegment blockTable = ws.getAttentionBlockTableSeg();
            MemorySegment contextLens = ws.getAttentionContextLensSeg();
            for (int b = 0; b < batchInt; b++) {
                for (int blk = 0; blk < maxBlocks; blk++) {
                    blockTable.setAtIndex(ValueLayout.JAVA_INT, (long) b * maxBlocks + blk, blk);
                }
            }

            for (int b = 0; b < batchInt; b++) {
                contextLens.setAtIndex(ValueLayout.JAVA_INT, b, contextTokens);
            }

            AccelTensor out = AccelTensor.zeros(q.shape());
            int result = numKVHeads == numHeads
                    ? metalBinding.attention(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap)
                    : metalBinding.attentionGqa(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, numKVHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap);
            if (result == 0) {
                DirectInferenceEngine.recordAttentionPath("sliding_decode_metal");
                return out;
            }
            out.close();
            DirectInferenceEngine.recordAttentionPath("sliding_decode_java");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
    }

    private AccelTensor denseCachedAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment kGathered = arena.allocate(gatherBytes, 64);
            MemorySegment vGathered = arena.allocate(gatherBytes, 64);
            gatherKV(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim, kGathered, vGathered);
            return denseCachedAttention(q, kGathered, vGathered, startPos, numHeads, numKVHeads, headDim, scale,
                    causal, softCap, config, layerIdx);
        }
    }

    private AccelTensor denseCachedAttention(AccelTensor q, MemorySegment kSeg, MemorySegment vSeg, int startPos,
            int numQHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx) {
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = startPos + (int) seqLenQ;
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        int slidingWindow = slidingLayer ? config.slidingWindowSize() : Integer.MAX_VALUE;
        AccelTensor out = AccelTensor.zeros(q.shape());

        MemorySegment qSeg = q.dataSegment();
        MemorySegment oSeg = out.dataSegment();
        long qStride0 = q.stride()[0];
        long qStride1 = q.stride()[1];
        long qStride2 = q.stride()[2];
        long oStride0 = out.stride()[0];
        long oStride1 = out.stride()[1];
        long oStride2 = out.stride()[2];
        int gqaGroup = numQHeads / Math.max(1, numKVHeads);

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numQHeads; h++) {
                int kvHeadIdx = h / gqaGroup;
                for (int i = 0; i < seqLenQ; i++) {
                    long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2);
                    float m = Float.NEGATIVE_INFINITY;
                    float l = 0.0f;
                    float[] acc = new float[headDim];
                    int minPos = slidingWindow == Integer.MAX_VALUE
                            ? 0
                            : Math.max(0, startPos + i - slidingWindow + 1);

                    for (int tok = 0; tok < totalTokens; tok++) {
                        if (tok < minPos) {
                            continue;
                        }
                        if (causal && tok > startPos + i) {
                            break;
                        }

                        long kOff = ((long) tok * numKVHeads + kvHeadIdx) * headDim;
                        float score = dotProduct(qSeg, qOff * 4L, kSeg, kOff * 4L, headDim) * scale;
                        if (softCap > 0.0f) {
                            score = (float) (Math.tanh(score / softCap) * softCap);
                        }

                        float mPrev = m;
                        m = Math.max(m, score);
                        float expPrev = (float) Math.exp(mPrev - m);
                        float expCurr = (float) Math.exp(score - m);
                        l = l * expPrev + expCurr;

                        long vOff = ((long) tok * numKVHeads + kvHeadIdx) * headDim;
                        updateAccumulator(acc, vSeg, vOff * 4L, expPrev, expCurr, headDim);
                    }

                    long oOff = ((long) b * oStride0 + (long) i * oStride1 + (long) h * oStride2);
                    for (int d = 0; d < headDim; d++) {
                        oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, oOff + d, acc[d] / (l + 1e-9f));
                    }
                }
            }
        }

        return out;
    }

    private void gatherKV(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int totalTokens, int numKVHeads, int headDim, MemorySegment kOut, MemorySegment vOut) {
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        int blockSize = kvSession.tokensPerBlock();
        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        for (int blk = 0; blk < blocks.size(); blk++) {
            int phys = blocks.get(blk);
            int tokenStart = blk * blockSize;
            int tokenEnd = Math.min(totalTokens, tokenStart + blockSize);
            MemorySegment kBlock = blockManager.getKBlock(phys);
            MemorySegment vBlock = blockManager.getVBlock(phys);
            MemorySegment kScaleBlock = blockManager.getKScaleBlock(phys);
            MemorySegment vScaleBlock = blockManager.getVScaleBlock(phys);

            for (int tok = tokenStart; tok < tokenEnd; tok++) {
                int tokInBlock = tok - tokenStart;
                for (int h = 0; h < numKVHeads; h++) {
                    long srcElement = ((long) h * blockManager.getHeadStride()) + ((long) tokInBlock * blockManager.getTokenStride());
                    long dstElement = ((long) tok * numKVHeads + h) * headDim;
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokInBlock;
                    writeVectorAsFloat(storageType, kBlock, kScaleBlock, srcElement, scaleIndex, kOut, dstElement, headDim);
                    writeVectorAsFloat(storageType, vBlock, vScaleBlock, srcElement, scaleIndex, vOut, dstElement, headDim);
                }
            }
        }
    }

    private void packDenseSharedKvIntoTemporaryPagedPool(AccelTensor key, AccelTensor value,
            int numKVHeads, int headDim, MemorySegment packedK, MemorySegment packedV) {
        long batch = key.size(0);
        int totalTokens = Math.toIntExact(key.size(1));
        MemorySegment keySeg = key.dataSegment();
        MemorySegment valueSeg = value.dataSegment();
        long keyStride0 = key.stride()[0];
        long keyStride1 = key.stride()[1];
        long keyStride2 = key.stride()[2];
        long valueStride0 = value.stride()[0];
        long valueStride1 = value.stride()[1];
        long valueStride2 = value.stride()[2];
        long perBatchBlockElements = (long) numKVHeads * totalTokens * headDim;

        for (int b = 0; b < batch; b++) {
            long batchBase = b * perBatchBlockElements;
            for (int tok = 0; tok < totalTokens; tok++) {
                for (int h = 0; h < numKVHeads; h++) {
                    long srcKElement = (long) b * keyStride0 + (long) tok * keyStride1 + (long) h * keyStride2;
                    long srcVElement = (long) b * valueStride0 + (long) tok * valueStride1 + (long) h * valueStride2;
                    long dstElement = batchBase + (long) h * totalTokens * headDim + (long) tok * headDim;
                    long bytes = (long) headDim * Float.BYTES;
                    MemorySegment.copy(keySeg, srcKElement * Float.BYTES, packedK, dstElement * Float.BYTES, bytes);
                    MemorySegment.copy(valueSeg, srcVElement * Float.BYTES, packedV, dstElement * Float.BYTES, bytes);
                }
            }
        }
    }

    private MaterializedKvPools materializeKvBlocksForMetal(BlockManager blockManager,
            KVCacheManager.KVCacheSession kvSession, List<Integer> blocks,
            int numKVHeads, int headDim, Arena arena) {
        int blockSize = kvSession.tokensPerBlock();
        long blockElements = (long) numKVHeads * blockSize * headDim;
        long totalElements = (long) blocks.size() * blockElements;
        MemorySegment packedK = arena.allocate(totalElements * Float.BYTES, 64);
        MemorySegment packedV = arena.allocate(totalElements * Float.BYTES, 64);
        BlockManager.KvStorageType storageType = blockManager.getStorageType();

        for (int localBlockIdx = 0; localBlockIdx < blocks.size(); localBlockIdx++) {
            int physBlockIdx = blocks.get(localBlockIdx);
            MemorySegment srcKBlock = blockManager.getKBlock(physBlockIdx);
            MemorySegment srcVBlock = blockManager.getVBlock(physBlockIdx);
            MemorySegment srcKScaleBlock = blockManager.getKScaleBlock(physBlockIdx);
            MemorySegment srcVScaleBlock = blockManager.getVScaleBlock(physBlockIdx);

            for (int h = 0; h < numKVHeads; h++) {
                for (int tok = 0; tok < blockSize; tok++) {
                    long srcElement = ((long) h * blockManager.getHeadStride()) + ((long) tok * blockManager.getTokenStride());
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tok;
                    long dstElement = ((long) localBlockIdx * blockElements)
                            + ((long) h * blockSize * headDim)
                            + ((long) tok * headDim);
                    writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock, srcElement, scaleIndex, packedK, dstElement,
                            headDim);
                    writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock, srcElement, scaleIndex, packedV, dstElement,
                            headDim);
                }
            }
        }

        return new MaterializedKvPools(packedK, packedV);
    }

    private void writeVectorAsFloat(BlockManager.KvStorageType storageType,
            MemorySegment srcBlock, MemorySegment scaleBlock,
            long srcElement, long scaleIndex,
            MemorySegment dst, long dstElement,
            int headDim) {
        switch (storageType) {
            case FP32 -> {
                long bytes = (long) headDim * Float.BYTES;
                MemorySegment.copy(srcBlock, srcElement * Float.BYTES, dst, dstElement * Float.BYTES, bytes);
            }
            case INT8 -> {
                float scale = scaleBlock == null ? 1.0f : scaleBlock.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
                for (int d = 0; d < headDim; d++) {
                    float value = srcBlock.getAtIndex(ValueLayout.JAVA_BYTE, srcElement + d) * scale;
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElement + d, value);
                }
            }
            case INT4 -> {
                float scale = scaleBlock == null ? 1.0f : scaleBlock.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
                for (int d = 0; d < headDim; d++) {
                    float value = readPackedSignedInt4(srcBlock, srcElement + d) * scale;
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstElement + d, value);
                }
            }
        }
    }

    private int readPackedSignedInt4(MemorySegment srcSeg, long srcElementIndex) {
        long byteIndex = srcElementIndex >>> 1;
        int packed = Byte.toUnsignedInt(srcSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int nibble = (srcElementIndex & 1L) == 0L ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return nibble - 8;
    }

    private record MaterializedKvPools(MemorySegment kPool, MemorySegment vPool) {
    }

    private void packKvRangeIntoTemporaryPagedPool(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession,
            int kvLayerIdx, int tokenStart, int tokenEnd, int numKVHeads, int headDim, int blockSize,
            MemorySegment packedK, MemorySegment packedV) {
        long srcHeadStride = blockManager.getHeadStride();
        long srcTokenStride = blockManager.getTokenStride();
        long dstBlockStride = (long) numKVHeads * blockSize * headDim;
        long dstHeadStride = (long) blockSize * headDim;
        BlockManager.KvStorageType storageType = blockManager.getStorageType();

        for (int tok = tokenStart; tok < tokenEnd; tok++) {
            int srcBlockIdx = kvSession.getBlockForToken(kvLayerIdx, tok);
            int srcTokInBlock = tok % kvSession.tokensPerBlock();
            int localTok = tok - tokenStart;
            int dstBlockIdx = localTok / blockSize;
            int dstTokInBlock = localTok % blockSize;

            MemorySegment srcKBlock = blockManager.getKBlock(srcBlockIdx);
            MemorySegment srcVBlock = blockManager.getVBlock(srcBlockIdx);
            MemorySegment srcKScaleBlock = blockManager.getKScaleBlock(srcBlockIdx);
            MemorySegment srcVScaleBlock = blockManager.getVScaleBlock(srcBlockIdx);
            for (int h = 0; h < numKVHeads; h++) {
                long srcElement = ((long) h * srcHeadStride) + ((long) srcTokInBlock * srcTokenStride);
                long scaleIndex = (long) h * blockManager.getScaleStride() + srcTokInBlock;
                long dstElement = ((long) dstBlockIdx * dstBlockStride) + ((long) h * dstHeadStride)
                        + ((long) dstTokInBlock * headDim);
                if (storageType == BlockManager.KvStorageType.FP32) {
                    long bytes = (long) headDim * Float.BYTES;
                    MemorySegment.copy(srcKBlock, srcElement * Float.BYTES, packedK, dstElement * Float.BYTES, bytes);
                    MemorySegment.copy(srcVBlock, srcElement * Float.BYTES, packedV, dstElement * Float.BYTES, bytes);
                } else {
                    writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock, srcElement, scaleIndex, packedK, dstElement,
                            headDim);
                    writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock, srcElement, scaleIndex, packedV, dstElement,
                            headDim);
                }
            }
        }
    }

    private boolean canUseMetal() {
        if (Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY)) {
            return false;
        }
        return metalReady;
    }

    private AccelTensor project(AccelTensor input, AccelTensor weight, AccelTensor bias, String profileKey,
                                ModelConfig config) {
        long t0 = System.nanoTime();
        AccelTensor projected = tryMetalHalfLinear(input, weight, bias, config);
        if (projected == null) {
            projected = tryMetalFloatLinear(input, weight, bias);
        }
        if (projected == null) {
            projected = AccelOps.linear(input, weight, bias);
        }
        DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
        return projected;
    }

    private LinearPair tryMetalHalfLinearPairMixed(AccelTensor input,
                                                   AccelTensor firstWeight,
                                                   AccelTensor firstBias,
                                                   AccelTensor secondWeight,
                                                   AccelTensor secondBias,
                                                   String profileKey,
                                                   ModelConfig config) {
        if (!METAL_MIXED_HALF_LINEAR_PAIR_ENABLED
                || !canUseExperimentalMetalLinear()
                || !metalBinding.supportsMatmulTransposedRightHalfPairMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !canUseMixedHalfPairWeight(firstWeight, config)
                || !canUseMixedHalfPairWeight(secondWeight, config)) {
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(config, firstWeight, secondWeight);
        AccelTensor firstMetalWeight = toMetalHalfWeight(firstWeight, nativeBf16Weights, config);
        AccelTensor secondMetalWeight = toMetalHalfWeight(secondWeight, nativeBf16Weights, config);
        if (firstMetalWeight == null || secondMetalWeight == null) {
            return null;
        }

        long[] inputShape = input.shape();
        long[] firstWeightShape = firstMetalWeight.shape();
        long[] secondWeightShape = secondMetalWeight.shape();
        if (inputShape.length < 2
                || firstWeightShape.length != 2
                || secondWeightShape.length != 2
                || firstWeightShape[1] != secondWeightShape[1]) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L || k != firstWeightShape[1]) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long[] firstOutputShape = inputShape.clone();
        firstOutputShape[firstOutputShape.length - 1] = firstWeightShape[0];
        long[] secondOutputShape = inputShape.clone();
        secondOutputShape[secondOutputShape.length - 1] = secondWeightShape[0];
        AccelTensor first = AccelTensor.zeros(firstOutputShape);
        AccelTensor second = AccelTensor.zeros(secondOutputShape);

        try {
            int rc = metalBinding.matmulTransposedRightHalfPairMixed(
                    first.dataPtr(),
                    second.dataPtr(),
                    contiguousInput.dataPtr(),
                    firstMetalWeight.dataPtr(),
                    secondMetalWeight.dataPtr(),
                    Math.toIntExact(rows),
                    Math.toIntExact(k),
                    Math.toIntExact(firstWeightShape[0]),
                    Math.toIntExact(secondWeightShape[0]),
                    1.0f,
                    0.0f,
                    nativeBf16Weights);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPairMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new LinearPair(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private LinearTriple tryMetalHalfLinearTripleMixed(AccelTensor input,
                                                       AccelTensor firstWeight,
                                                       AccelTensor firstBias,
                                                       AccelTensor secondWeight,
                                                       AccelTensor secondBias,
                                                       AccelTensor thirdWeight,
                                                       AccelTensor thirdBias,
                                                       String profileKey,
                                                       ModelConfig config) {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_PROPERTY)
                || !METAL_MIXED_HALF_LINEAR_PAIR_ENABLED
                || !canUseExperimentalMetalLinear()
                || !metalBinding.supportsMatmulTransposedRightHalfTripleMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !canUseMixedHalfPairWeight(firstWeight, config)
                || !canUseMixedHalfPairWeight(secondWeight, config)
                || !canUseMixedHalfPairWeight(thirdWeight, config)) {
            return null;
        }

        long[] inputShape = input.shape();
        if (inputShape.length < 2) {
            return null;
        }
        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        boolean nativeBf16Weights = shouldUseNativeMetalBf16Linear(config, firstWeight, secondWeight, thirdWeight);
        AccelTensor firstMetalWeight = toMetalHalfWeight(firstWeight, nativeBf16Weights, config);
        AccelTensor secondMetalWeight = toMetalHalfWeight(secondWeight, nativeBf16Weights, config);
        AccelTensor thirdMetalWeight = toMetalHalfWeight(thirdWeight, nativeBf16Weights, config);
        if (firstMetalWeight == null || secondMetalWeight == null || thirdMetalWeight == null) {
            return null;
        }

        long[] firstWeightShape = firstMetalWeight.shape();
        long[] secondWeightShape = secondMetalWeight.shape();
        long[] thirdWeightShape = thirdMetalWeight.shape();
        if (firstWeightShape.length != 2
                || secondWeightShape.length != 2
                || thirdWeightShape.length != 2
                || firstWeightShape[1] != k
                || secondWeightShape[1] != k
                || thirdWeightShape[1] != k) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        long[] firstOutputShape = inputShape.clone();
        firstOutputShape[firstOutputShape.length - 1] = firstWeightShape[0];
        long[] secondOutputShape = inputShape.clone();
        secondOutputShape[secondOutputShape.length - 1] = secondWeightShape[0];
        long[] thirdOutputShape = inputShape.clone();
        thirdOutputShape[thirdOutputShape.length - 1] = thirdWeightShape[0];
        AccelTensor first = AccelTensor.zeros(firstOutputShape);
        AccelTensor second = AccelTensor.zeros(secondOutputShape);
        AccelTensor third = AccelTensor.zeros(thirdOutputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n0 = Math.toIntExact(firstWeightShape[0]);
            int n1 = Math.toIntExact(secondWeightShape[0]);
            int n2 = Math.toIntExact(thirdWeightShape[0]);
            int rc = -2;
            if (m == 1
                    && nativeBf16Weights
                    && shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightBf16TripleMixed()) {
                rc = metalBinding.matvecTransposedRightBf16TripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weights
                    && shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightHalfTripleMixed()) {
                rc = metalBinding.matvecTransposedRightHalfTripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfTripleMixed(
                    first.dataPtr(),
                    second.dataPtr(),
                    third.dataPtr(),
                    contiguousInput.dataPtr(),
                    firstMetalWeight.dataPtr(),
                    secondMetalWeight.dataPtr(),
                    thirdMetalWeight.dataPtr(),
                    m,
                    kk,
                    n0,
                    n1,
                    n2,
                    1.0f,
                    0.0f,
                    nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfTripleMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            AccelTensor thirdOut = addBiasIfNeeded(third, thirdBias);
            DirectInferenceEngine.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new LinearTriple(firstOut, secondOut, thirdOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            third.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private boolean canUseMixedHalfPairWeight(AccelTensor weight, ModelConfig config) {
        if (weight == null || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && isGemma4Text(config)) {
            return shouldUseNativeMetalBf16Linear(config, weight);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(config));
    }

    private static boolean resolveMetalMixedHalfLinearPairEnabled() {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_MIXED_HALF_LINEAR_PAIR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return true;
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
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(config)) {
            return weight.toF16CachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
        }
        return null;
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

    private record LinearTriple(AccelTensor first, AccelTensor second, AccelTensor third) {
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
        return true;
    }

    private boolean allowMetalBf16Linear(ModelConfig config) {
        if (isGemma4Text(config)) {
            if (Boolean.getBoolean(DISABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY)) {
                return false;
            }
            return allowMetalBf16Linear();
        }
        return allowMetalBf16Linear();
    }

    private boolean preferNativeMetalBf16Linear(ModelConfig config) {
        if (isGemma4Text(config)) {
            String explicit = System.getProperty(ENABLE_GEMMA4_METAL_BF16_LINEAR_PROPERTY);
            if (explicit != null && !explicit.isBlank()) {
                return Boolean.parseBoolean(explicit) && allowMetalBf16Linear(config);
            }
            return allowMetalBf16Linear(config);
        }
        return false;
    }

    private boolean shouldUseNativeMetalBf16Linear(ModelConfig config, AccelTensor... weights) {
        if (!preferNativeMetalBf16Linear(config) || weights == null || weights.length == 0) {
            return false;
        }
        for (AccelTensor weight : weights) {
            if (weight == null || weight.quantType() != AccelTensor.QuantType.BF16) {
                return false;
            }
        }
        return true;
    }

    private static boolean resolveAutoMetalHalfMatvecEnabled() {
        String explicit = System.getProperty(AUTO_METAL_ATTENTION_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        explicit = System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        return false;
    }

    private AccelTensor tryMetalHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
                                           ModelConfig config) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        boolean nativeBf16Weight = shouldUseNativeMetalBf16Linear(config, weight);
        if (quantType == AccelTensor.QuantType.BF16 && isGemma4Text(config) && !nativeBf16Weight) {
            return null;
        }
        if (quantType != AccelTensor.QuantType.F16
                && (quantType != AccelTensor.QuantType.BF16 || !allowMetalBf16Linear(config))) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }
        long[] inputShape = input.shape();
        long[] weightShape = weight.shape();
        if (inputShape.length < 2 || weightShape.length != 2) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = weightShape[0];
        AccelTensor out = AccelTensor.zeros(outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(weightShape[0]);
            int rc = -2;
            if (m == 1
                    && nativeBf16Weight
                    && shouldUseMetalHalfMatvec(config, n)
                    && metalBinding.supportsMatvecTransposedRightBf16()) {
                rc = metalBinding.matvecTransposedRightBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        weight.dataPtr(),
                        kk, n);
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && shouldUseMetalTransposedHalfMatvec(n)
                    && metalBinding.supportsMatvecTransposedWeightHalf()) {
                AccelTensor transposedWeight = weight.toF16Transposed2dCachedUpTo(METAL_F16_WEIGHT_CACHE_MAX_BYTES);
                if (transposedWeight != null
                        && transposedWeight.size(0) == k
                        && transposedWeight.size(1) == weightShape[0]) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                }
            }
            AccelTensor metalWeight = null;
            if (rc != 0) {
                metalWeight = toMetalHalfWeight(weight, nativeBf16Weight, config);
                if (metalWeight == null) {
                    return null;
                }
            }
            if (m == 1
                    && !nativeBf16Weight
                    && shouldUseMetalHalfMatvec(config, n)
                    && metalBinding.supportsMatvecTransposedRightHalf()) {
                rc = metalBinding.matvecTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
            }
            if (rc != 0 && metalWeight != null) {
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
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor tryMetalFloatLinear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        if (input == null || weight == null) {
            return null;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }

        long[] inputShape = input.shape();
        long[] weightShape = weight.shape();
        if (inputShape.length < 2 || weightShape.length != 2) {
            return null;
        }

        long k = inputShape[inputShape.length - 1];
        long rows = input.numel() / Math.max(1L, k);
        if (rows <= 0L) {
            return null;
        }

        long batchProduct = 1L;
        for (int i = 0; i < inputShape.length - 2; i++) {
            batchProduct *= inputShape[i];
        }
        if (batchProduct != 1L) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        long[] outputShape = inputShape.clone();
        outputShape[outputShape.length - 1] = weightShape[0];
        AccelTensor out = AccelTensor.zeros(outputShape);

        try {
            int m = Math.toIntExact(rows);
            int kk = Math.toIntExact(k);
            int n = Math.toIntExact(weightShape[0]);
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
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private boolean canUseMetalAttention(ModelConfig config, int layerIdx, KVCacheManager.KVCacheSession kvSession,
            int seqLen, int startPos, float softCap) {
        if (!canUseMetal()) {
            return false;
        }
        boolean gemma4Text = isGemma4Text(config);
        if (!allowLegacyMetalAttentionBridge(config) && gemma4Text) {
            return canUseGemma4SlidingPrefillFa4Attention(config, layerIdx, seqLen, startPos, softCap);
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            return metalBinding != null && metalBinding.isWindowedAttentionAvailable();
        }
        if (gemma4Text
                && Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY)
                && !Boolean.getBoolean(ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY)) {
            return false;
        }
        return true;
    }

    private boolean canUseGemma4SlidingPrefillFa4Attention(ModelConfig config, int layerIdx,
            int seqLen, int startPos, float softCap) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (seqLen <= 1 || config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (Boolean.getBoolean(DISABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY)
                || Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_GEMMA4_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY);
        if (explicit != null && !explicit.isBlank() && !Boolean.parseBoolean(explicit)) {
            return false;
        }
        int totalTokens = startPos + seqLen;
        return totalTokens <= config.slidingWindowSize() && canUseFa4Attention(softCap);
    }

    private boolean canUseFa4PagedAttention(ModelConfig config, int layerIdx,
            KVCacheManager.KVCacheSession kvSession, float softCap) {
        if (!canUseMetal() || !canUseFa4Attention(softCap)) {
            return false;
        }
        if (metalFa4 == null || !metalFa4.isNativeAvailable()) {
            return false;
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            return false;
        }
        return true;
    }

    private boolean canUseSlidingDecodeMetalAttention(ModelConfig config, int layerIdx,
            KVCacheManager.KVCacheSession kvSession, int seqLen) {
        if (!canUseMetal()) {
            return false;
        }
        if (!allowSlidingDecodeMetalAttentionBridge(config, layerIdx, seqLen)) {
            return false;
        }
        if (seqLen != 1) {
            return false;
        }
        if (metalBinding == null || !metalBinding.isNativeAvailable()) {
            return false;
        }
        if (config.slidingWindowSize() > MAX_PACKED_METAL_SLIDING_WINDOW) {
            return false;
        }
        if (isGemma4Text(config)
                && Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY)
                && !Boolean.getBoolean(ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY)) {
            return false;
        }
        return !config.usesSharedKvCache(layerIdx);
    }

    private boolean allowSlidingDecodeMetalAttentionBridge(ModelConfig config, int layerIdx, int seqLen) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (allowLegacyMetalAttentionBridge(config)) {
            return true;
        }
        if (!isGemma4Text(config) || seqLen != 1) {
            return false;
        }
        if (config.slidingWindowSize() > MAX_PACKED_METAL_SLIDING_WINDOW) {
            return false;
        }
        return !Boolean.getBoolean(DISABLE_METAL_GEMMA4_ATTENTION_PROPERTY)
                || Boolean.getBoolean(ALLOW_METAL_GEMMA4_ATTENTION_PROPERTY);
    }

    private boolean canUseDenseGemma4Attention(ModelConfig config, int layerIdx) {
        if (!isGemma4Text(config) || !Boolean.getBoolean(FORCE_DENSE_GEMMA4_ATTENTION_PROPERTY)) {
            return false;
        }
        return config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow();
    }

    private boolean canUseFa4Attention(float softCap) {
        return metalFa4 != null
                && metalFa4.isNativeAvailable();
    }

    private boolean preferPagedMetalAttentionBeforeFa4(ModelConfig config, int seqLen, int totalTokens) {
        if (!allowPagedMetalAttentionBridge(config, seqLen, totalTokens)) {
            return false;
        }
        int maxTokens = Integer.getInteger(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY,
                DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS);
        return maxTokens > 0 && seqLen == 1 && totalTokens <= maxTokens;
    }

    private boolean allowPagedMetalAttentionBridge(ModelConfig config, int seqLen, int totalTokens) {
        if (allowLegacyMetalAttentionBridge(config)) {
            return true;
        }
        if (!isGemma4Text(config) || seqLen != 1) {
            return false;
        }
        if (Boolean.getBoolean(DISABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY)) {
            return false;
        }
        if (Boolean.getBoolean(ENABLE_GEMMA4_PAGED_DECODE_ATTENTION_PROPERTY)) {
            return true;
        }
        int maxTokens = Integer.getInteger(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY,
                DEFAULT_PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS);
        return maxTokens > 0 && totalTokens <= maxTokens;
    }

    private boolean allowLegacyMetalAttentionBridge(ModelConfig config) {
        if (!isGemma4Text(config)) {
            return true;
        }
        return Boolean.getBoolean(ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY);
    }

    private boolean shouldUseMetalHalfMatvec(ModelConfig config, int outputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY);
        int maxOutput = Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit) && outputDim <= maxOutput;
        }
        if (isGemma4Text(config)) {
            return outputDim <= maxOutput;
        }
        return AUTO_METAL_HALF_MATVEC_ENABLED
                && outputDim <= maxOutput
                && isMetalHalfMatvecAutoCandidate(config);
    }

    private static boolean shouldUseMetalTransposedHalfMatvec(int outputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY);
        boolean enabled = explicit != null && !explicit.isBlank() && Boolean.parseBoolean(explicit);
        int maxOutput = Integer.getInteger(METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT);
        return enabled && outputDim <= maxOutput;
    }

    private static boolean shouldUseMetalHalfTripleMatvec(int firstOutputDim, int secondOutputDim, int thirdOutputDim) {
        if (Boolean.getBoolean(DISABLE_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_PROPERTY)) {
            return false;
        }
        int maxOutput = Integer.getInteger(METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT_PROPERTY,
                DEFAULT_METAL_MIXED_HALF_LINEAR_TRIPLE_MATVEC_MAX_OUTPUT);
        return firstOutputDim > 0
                && secondOutputDim > 0
                && thirdOutputDim > 0
                && firstOutputDim + secondOutputDim + thirdOutputDim <= maxOutput;
    }

    private boolean isMetalHalfMatvecAutoCandidate(ModelConfig config) {
        if (config == null || config.modelType() == null || config.modelType().isBlank()) {
            return false;
        }
        String modelType = config.modelType().toLowerCase();
        if (isGemma4Text(config) || hasGemma4StylePerLayerInputs(config)) {
            return false;
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

    private float resolveAttentionSoftCap(ModelArchitecture arch, ModelConfig config, boolean gemma4Text) {
        if (gemma4Text) {
            // Gemma-4 text soft-caps final logits, not attention scores.
            return 0.0f;
        }
        Double configured = config != null ? config.attnLogitSoftcapping() : null;
        if (configured != null && configured > 0) {
            return configured.floatValue();
        }
        return arch.defaultAttnSoftCap();
    }
}
