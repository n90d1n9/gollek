package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.metal.MetalComputeBackend;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import jakarta.enterprise.inject.Instance;

@ApplicationScoped
public class FlashAttentionKernel {
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";

    @Inject
    Instance<MetalComputeBackend> metalBackendInstance;
    
    private MetalComputeBackend metal;
    private MetalFlashAttentionBinding metalFa4;
    private MetalBinding metalBinding;
    
    @Inject
    RopeFrequencyCache ropeCache;

    @jakarta.annotation.PostConstruct
    void init() {
        if (metalBackendInstance.isResolvable()) {
            this.metal = metalBackendInstance.get();
        }
        this.metalBinding = MetalBinding.getInstance();
        MetalFlashAttentionBinding.initialize();
        this.metalFa4 = MetalFlashAttentionBinding.getInstance();
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

        public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
                AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
                ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
                int layerIdx, int startPos, boolean isCausal,
                AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW) {
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
        }
    }

    public AccelTensor compute(AttentionInput in) {
        ModelConfig config = in.config;
        int layerIdx = in.layerIdx;
        int startPos = in.startPos;
        int seqLen = (int) in.x.size(1);
        int numQHeads = config.numAttentionHeads();
        int headDim = (int) in.qW.size(0) / numQHeads;
        int numKVHeads = config.resolvedNumKvHeads();
        boolean gemma4Text = isGemma4Text(config);
        float scale = gemma4Text ? 1.0f : (float) (1.0 / Math.sqrt(headDim));
        float attnSoftCap = gemma4Text ? 0.0f : in.arch.defaultAttnSoftCap();
        boolean sharedKv = config.usesSharedKvCache(layerIdx);
        int kvLayerIdx = config.sharedKvSourceLayer(layerIdx);

        // 1. Projections
        AccelTensor q = AccelOps.linear(in.x, in.qW, in.qB);
        AccelTensor k = sharedKv ? null : AccelOps.linear(in.x, in.kW, in.kB);
        AccelTensor v = sharedKv ? null : AccelOps.linear(in.x, in.vW, in.vB);

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
        if (in.qNormW != null) {
            AccelTensor qNormed = AccelOps.perHeadRmsNorm(q, in.qNormW, config.rmsNormEps(), addOneRmsNorm);
            q.close();
            q = qNormed;
        }
        if (!sharedKv && in.kNormW != null) {
            AccelTensor kNormed = AccelOps.perHeadRmsNorm(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm);
            k.close();
            k = kNormed;
        }

        if (!sharedKv) {
            AccelTensor v4 = v.reshape(in.x.size(0), in.x.size(1), numKVHeads, headDim);
            v.close();
            v = v4;
            if (gemma4Text) {
                AccelTensor vNormed = perHeadRmsNormNoWeight(v, config.rmsNormEps());
                v.close();
                v = vNormed;
            }
        }

        int rotaryDim = resolveRotaryDim(config, layerIdx, headDim);
        RopeFrequencyCache.RopeFrequencies freqs = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(),
                config.ropeThetaForLayer(layerIdx), config.ropeScaling());
        applyRope(q, sharedKv ? null : k, startPos, freqs, !in.arch.usesNeoxRope());

        // 4. Update Cache
        KVCacheManager.KVCacheSession kvSession = in.kvCache;
        if (!sharedKv) {
            updateKVCache(k, v, kvSession, layerIdx, startPos, seqLen, numKVHeads, headDim);
        }

        // 5. Attention
        AccelTensor attnOut;
        if (canUseMetal() && !kvSession.isQuantizedInt8()) {
            attnOut = tiledAttention(q, kvSession, kvLayerIdx, startPos, numQHeads, numKVHeads, headDim, scale, in.isCausal,
                    attnSoftCap);
        } else {
            attnOut = PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numQHeads, headDim, scale,
                    in.isCausal, attnSoftCap);
        }
        q.close();
        if (k != null) k.close();
        if (v != null) v.close();

        // Reshape back to 3D for subsequent operations
        AccelTensor attnOut3 = attnOut.reshape(in.x.size(0), seqLen, numQHeads * headDim);
        attnOut.close();
        attnOut = attnOut3;

        // 6. Output Projection: maps [B,T,numQHeads*headDim] → [B,T,hidden_size]
        AccelTensor projected = AccelOps.linear(attnOut, in.oW, in.oB);
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

    private AccelTensor rmsNorm(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        return AccelOps.rmsNorm(x, w, eps, addOne);
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

    private int resolveRotaryDim(ModelConfig config, int layerIdx, int headDim) {
        double partialFactor = config.partialRotaryFactorForLayer(layerIdx);
        int rotaryDim = (int) Math.round(headDim * partialFactor);
        rotaryDim = Math.max(2, rotaryDim);
        if ((rotaryDim & 1) != 0) {
            rotaryDim--;
        }
        return Math.min(headDim, rotaryDim);
    }

    private boolean isGemma4Text(ModelConfig config) {
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        return modelType.startsWith("gemma4");
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
                float sumSq = 0.0f;
                for (int i = 0; i < headDim; i++) {
                    float val = xSeg.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
                    sumSq += val * val;
                }
                float rms = (float) (1.0 / Math.sqrt(sumSq / headDim + eps));
                for (int j = 0; j < headDim; j++) {
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

    private AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        java.util.List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);

        try (Arena arena = Arena.ofConfined()) {
            AccelTensor out = AccelTensor.zeros(q.shape());
            if (metalFa4 != null && metalFa4.isNativeAvailable() && metalFa4.isSdpaAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                gatherKV(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim, kGathered, vGathered);

                boolean useBf16 = metalFa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = metalFa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16);
                if (result == 0) {
                    return out;
                }
                out.close();
                out = AccelTensor.zeros(q.shape());
            }
            if (metalBinding != null && metalBinding.isNativeAvailable() && blocks != null && !blocks.isEmpty()) {
                int maxBlocks = blocks.size();
                int[] blockTable = new int[(int) batch * maxBlocks];
                for (int b = 0; b < (int) batch; b++) {
                    int base = b * maxBlocks;
                    for (int i = 0; i < maxBlocks; i++) {
                        blockTable[base + i] = blocks.get(i);
                    }
                }
                int[] contextLens = new int[(int) batch];
                java.util.Arrays.fill(contextLens, totalTokens);
                MemorySegment blockTableSegment = arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_INT, blockTable);
                MemorySegment contextLensSegment = arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_INT, contextLens);

                int result = numKVHeads == numHeads
                        ? metalBinding.attention(
                                out.dataPtr(), q.dataPtr(), kvSession.getRawKPool(), kvSession.getRawVPool(),
                                blockTableSegment, contextLensSegment,
                                (int) batch, (int) seqLen, numHeads, headDim,
                                kvSession.tokensPerBlock(), maxBlocks,
                                scale, causal ? 1 : 0, softCap)
                        : metalBinding.attentionGqa(
                                out.dataPtr(), q.dataPtr(), kvSession.getRawKPool(), kvSession.getRawVPool(),
                                blockTableSegment, contextLensSegment,
                                (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                kvSession.tokensPerBlock(), maxBlocks,
                                scale, causal ? 1 : 0, softCap);
                if (result == 0) {
                    return out;
                }
                out.close();
                out = AccelTensor.zeros(q.shape());
            }
            if (metalFa4 != null && metalFa4.isNativeAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                gatherKV(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim, kGathered, vGathered);

                boolean useBf16 = metalFa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = metalFa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16);
                if (result == 0) {
                    return out;
                }
                out.close();
            }
            return PagedAttentionVectorAPI.compute(q, null, kvSession, kvLayerIdx, kvLayerIdx, startPos, numHeads, headDim, scale, causal, softCap);
        }
    }

    private void gatherKV(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int totalTokens, int numKVHeads, int headDim, MemorySegment kOut, MemorySegment vOut) {
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        int blockSize = kvSession.tokensPerBlock();
        for (int blk = 0; blk < blocks.size(); blk++) {
            int phys = blocks.get(blk);
            int tokenStart = blk * blockSize;
            int tokenEnd = Math.min(totalTokens, tokenStart + blockSize);
            MemorySegment kBlock = blockManager.getKBlock(phys);
            MemorySegment vBlock = blockManager.getVBlock(phys);

            for (int tok = tokenStart; tok < tokenEnd; tok++) {
                int tokInBlock = tok - tokenStart;
                for (int h = 0; h < numKVHeads; h++) {
                    long srcElement = ((long) h * blockManager.getHeadStride()) + ((long) tokInBlock * blockManager.getTokenStride());
                    long dstElement = ((long) tok * numKVHeads + h) * headDim;
                    long bytes = (long) headDim * Float.BYTES;
                    MemorySegment.copy(kBlock, srcElement * Float.BYTES, kOut, dstElement * Float.BYTES, bytes);
                    MemorySegment.copy(vBlock, srcElement * Float.BYTES, vOut, dstElement * Float.BYTES, bytes);
                }
            }
        }
    }

    private boolean canUseMetal() {
        if (Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY)) {
            return false;
        }
        return metal != null && metal.deviceName() != null && !metal.deviceName().contains("CPU");
    }
}
