package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

/**
 * Paged Attention-2 implementation using Java scalar loops (fallback).
 * Traverses a block-mapped KV cache for memory-efficient inference.
 * Matches the layout: [numHeads, tokensPerBlock, headDim]
 */
public class PagedAttentionVectorAPI {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final String DEBUG_ATTENTION_PROBE_PROPERTY = "gollek.debug.attention_probe";

    /**
     * Compute attention using the Paged Attention algorithm.
     */
    public static AccelTensor compute(
            AccelTensor q,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvSession,
            int layerIdx,
            int kvLayerIdx,
            int startPos,
            int numQHeads,
            int numKVHeads,
            int headDim,
            float scale,
            boolean causal,
            float softCap) {

        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = startPos + (int) seqLenQ;
        int tokensPerBlock = kvSession.tokensPerBlock();
        List<Integer> blockTable = kvSession.getBlockIndices(kvLayerIdx);
        BlockManager blockManager = kvSession.blockManager();
        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        int slidingWindow = slidingLayer ? config.slidingWindowSize() : Integer.MAX_VALUE;
        boolean debugProbe = Boolean.getBoolean(DEBUG_ATTENTION_PROBE_PROPERTY);

        AccelTensor out = AccelTensor.zeros(q.shape());
        float[] accScratch = new float[headDim];

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numQHeads; h++) {
                for (int i = 0; i < seqLenQ; i++) {
                    computePagedHeadQuery(b, h, i, q, blockTable, blockManager, out, scale, causal, tokensPerBlock,
                            totalTokens, numKVHeads, headDim, softCap, storageType, slidingWindow, accScratch,
                            debugProbe && layerIdx == 0 && b == 0 && h == 0 && i == seqLenQ - 1);
                }
            }
        }
        return out;
    }

    private static void computePagedHeadQuery(int b, int h, int i,
            AccelTensor q, List<Integer> blockTable, BlockManager blockManager, AccelTensor out,
            float scale, boolean causal, int tokensPerBlock, int totalTokens, int numKVHeads, int headDim,
            float softCap, BlockManager.KvStorageType storageType, int slidingWindow, float[] acc,
            boolean debugProbe) {

        MemorySegment qSeg = q.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        long qStride0 = q.stride()[0];
        long qStride1 = q.stride()[1];
        long qStride2 = q.stride()[2];

        // Correct offset accounting for tensor strides
        long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2) * 4;

        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Arrays.fill(acc, 0, headDim, 0.0f);

        int qHeads = (int)q.size(2);
        int gqaGroup = qHeads / numKVHeads;
        int kvHeadIdx = h / gqaGroup;
        int seqLenQ = (int) q.size(1);
        int queryStartPos = totalTokens - seqLenQ;
        int queryAbsPos = queryStartPos + i;
        int minPos = slidingWindow == Integer.MAX_VALUE ? 0 : Math.max(0, queryAbsPos - slidingWindow + 1);

        long headStride = blockManager.getHeadStride();
        long tokenStride = blockManager.getTokenStride();

        // Softmax Online Loop over blocks
        int currentTokCount = 0;
        for (int blockIdx : blockTable) {
            MemorySegment kBlock = blockManager.getKBlock(blockIdx);
            MemorySegment vBlock = blockManager.getVBlock(blockIdx);
            MemorySegment kScaleBlock = blockManager.getKScaleBlock(blockIdx);
            MemorySegment vScaleBlock = blockManager.getVScaleBlock(blockIdx);
            int tokensInThisBlock = Math.min(tokensPerBlock, totalTokens - currentTokCount);

            for (int tok = 0; tok < tokensInThisBlock; tok++) {
                int absPos = currentTokCount + tok;
                if (absPos < minPos) {
                    continue;
                }
                if (causal && absPos > queryAbsPos)
                    break;

                // 1. Compute Dot Product Q_i @ K_j
                long kvOff = ((long) kvHeadIdx * headStride + (long) tok * tokenStride) * 4;
                long kvElementOff = ((long) kvHeadIdx * headStride + (long) tok * tokenStride);
                long scaleIndex = (long) kvHeadIdx * blockManager.getScaleStride() + tok;
                float score = switch (storageType) {
                    case INT8 -> dotProductInt8(qSeg, qOff, kBlock, kvElementOff, kScaleBlock, scaleIndex, headDim) * scale;
                    case INT4 -> dotProductInt4(qSeg, qOff, kBlock, kvElementOff, kScaleBlock, scaleIndex, headDim) * scale;
                    case FP32 -> dotProduct(qSeg, qOff, kBlock, kvOff, headDim) * scale;
                };
                
                if (softCap > 0.0f) {
                    score = (float) (Math.tanh(score / softCap) * softCap);
                }
                if (debugProbe && absPos < 12) {
                    System.err.printf("[DEBUG-ATTN] layer=0 head=0 query=%d key=%d score=%f%n",
                            queryAbsPos, absPos, score);
                }

                // 2. Online Softmax Update
                float m_prev = m;
                m = Math.max(m, score);
                float exp_prev = (float) Math.exp(m_prev - m);
                float exp_curr = (float) Math.exp(score - m);
                l = l * exp_prev + exp_curr;

                // 3. Update Output Accumulator
                switch (storageType) {
                    case INT8 -> updateAccumulatorInt8(acc, vBlock, kvElementOff, vScaleBlock, scaleIndex, exp_prev, exp_curr, headDim);
                    case INT4 -> updateAccumulatorInt4(acc, vBlock, kvElementOff, vScaleBlock, scaleIndex, exp_prev, exp_curr, headDim);
                    case FP32 -> updateAccumulator(acc, vBlock, kvOff, exp_prev, exp_curr, headDim);
                }
            }

            currentTokCount += tokensInThisBlock;
            if (currentTokCount >= totalTokens)
                break;
        }

        // 4. Final Normalization
        long oStride0 = out.stride()[0];
        long oStride1 = out.stride()[1];
        long oStride2 = out.stride()[2];
        long oOff = ((long) b * oStride0 + (long) i * oStride1 + (long) h * oStride2) * 4;

        long outIndex = oOff / Float.BYTES;
        float invL = 1.0f / (l + 1e-9f);
        writeNormalizedAccumulator(oSeg, outIndex, acc, invL, headDim);
    }

    private static float dotProduct(MemorySegment q, long qOff, MemorySegment k, long kOff, long dim) {
        int n = Math.toIntExact(dim);
        int j = 0;
        FloatVector sum = FloatVector.zero(FLOAT_SPECIES);
        int upperBound = FLOAT_SPECIES.loopBound(n);
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector qVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, q, qOff + (long) j * Float.BYTES, ByteOrder.nativeOrder());
            FloatVector kVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, k, kOff + (long) j * Float.BYTES, ByteOrder.nativeOrder());
            sum = sum.add(qVec.mul(kVec));
        }
        float res = sum.reduceLanes(VectorOperators.ADD);
        for (; j < n; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + j) *
                    k.getAtIndex(ValueLayout.JAVA_FLOAT, (kOff / 4) + j);
        }
        return res;
    }

    private static void updateAccumulator(float[] acc, MemorySegment vSeg, long vOff, float exp_prev, float exp_curr,
            long dim) {
        int n = Math.toIntExact(dim);
        int j = 0;
        FloatVector prev = FloatVector.broadcast(FLOAT_SPECIES, exp_prev);
        FloatVector curr = FloatVector.broadcast(FLOAT_SPECIES, exp_curr);
        int upperBound = FLOAT_SPECIES.loopBound(n);
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector accVec = FloatVector.fromArray(FLOAT_SPECIES, acc, j);
            FloatVector valueVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, vSeg, vOff + (long) j * Float.BYTES, ByteOrder.nativeOrder());
            accVec.mul(prev).add(valueVec.mul(curr)).intoArray(acc, j);
        }
        for (; j < n; j++) {
            acc[j] = acc[j] * exp_prev + vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (vOff / 4) + j) * exp_curr;
        }
    }

    private static void writeNormalizedAccumulator(MemorySegment out, long outIndex, float[] acc, float invL,
            int headDim) {
        int j = 0;
        FloatVector inv = FloatVector.broadcast(FLOAT_SPECIES, invL);
        int upperBound = FLOAT_SPECIES.loopBound(headDim);
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, acc, j)
                    .mul(inv)
                    .intoMemorySegment(out, (outIndex + j) * Float.BYTES, ByteOrder.nativeOrder());
        }
        for (; j < headDim; j++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, outIndex + j, acc[j] * invL);
        }
    }

    private static float dotProductInt8(MemorySegment q, long qOff, MemorySegment k, long kElementOff,
            MemorySegment scaleSeg, long scaleIndex, long dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        float res = 0.0f;
        for (int j = 0; j < dim; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + j)
                    * (k.getAtIndex(ValueLayout.JAVA_BYTE, kElementOff + j) * scale);
        }
        return res;
    }

    private static void updateAccumulatorInt8(float[] acc, MemorySegment vSeg, long vElementOff,
            MemorySegment scaleSeg, long scaleIndex, float exp_prev, float exp_curr, long dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        for (int j = 0; j < dim; j++) {
            acc[j] = acc[j] * exp_prev + (vSeg.getAtIndex(ValueLayout.JAVA_BYTE, vElementOff + j) * scale) * exp_curr;
        }
    }

    private static float dotProductInt4(MemorySegment q, long qOff, MemorySegment k, long kElementOff,
            MemorySegment scaleSeg, long scaleIndex, long dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        float res = 0.0f;
        for (int j = 0; j < dim; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + j)
                    * (readPackedSignedInt4(k, kElementOff + j) * scale);
        }
        return res;
    }

    private static void updateAccumulatorInt4(float[] acc, MemorySegment vSeg, long vElementOff,
            MemorySegment scaleSeg, long scaleIndex, float exp_prev, float exp_curr, long dim) {
        float scale = scaleSeg == null ? 1.0f : scaleSeg.getAtIndex(ValueLayout.JAVA_FLOAT, scaleIndex);
        for (int j = 0; j < dim; j++) {
            acc[j] = acc[j] * exp_prev + (readPackedSignedInt4(vSeg, vElementOff + j) * scale) * exp_curr;
        }
    }

    private static int readPackedSignedInt4(MemorySegment segment, long elementIndex) {
        long byteIndex = elementIndex >>> 1;
        int packed = Byte.toUnsignedInt(segment.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int nibble = (elementIndex & 1L) == 0L ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return nibble - 8;
    }
}
