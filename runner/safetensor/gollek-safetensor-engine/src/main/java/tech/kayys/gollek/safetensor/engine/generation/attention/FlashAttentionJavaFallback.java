/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

final class FlashAttentionJavaFallback {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private FlashAttentionJavaFallback() {
    }

    static AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
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
                float[] acc = new float[headDim];
                for (int i = 0; i < seqLenQ; i++) {
                    long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2);
                    float m = Float.NEGATIVE_INFINITY;
                    float l = 0.0f;
                    Arrays.fill(acc, 0, headDim, 0.0f);
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
                    writeNormalizedAccumulator(oSeg, oOff, acc, 1.0f / (l + 1e-9f), headDim);
                }
            }
        }

        return out;
    }

    static AccelTensor denseCachedAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, int layerIdx) {
        BlockManager blockManager = kvSession.blockManager();
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment kGathered = arena.allocate(gatherBytes, 64);
            MemorySegment vGathered = arena.allocate(gatherBytes, 64);
            PagedKvCacheIO.gather(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim,
                    kGathered, vGathered);
            return denseCachedAttention(q, kGathered, vGathered, startPos, numHeads, numKVHeads, headDim, scale,
                    causal, softCap, config, layerIdx);
        }
    }

    private static AccelTensor denseCachedAttention(AccelTensor q, MemorySegment kSeg, MemorySegment vSeg,
            int startPos, int numQHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
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
                float[] acc = new float[headDim];
                for (int i = 0; i < seqLenQ; i++) {
                    long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2);
                    float m = Float.NEGATIVE_INFINITY;
                    float l = 0.0f;
                    Arrays.fill(acc, 0, headDim, 0.0f);
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
                    writeNormalizedAccumulator(oSeg, oOff, acc, 1.0f / (l + 1e-9f), headDim);
                }
            }
        }

        return out;
    }

    private static float dotProduct(MemorySegment q, long qOffBytes, MemorySegment k, long kOffBytes, int headDim) {
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

    private static void updateAccumulator(float[] acc, MemorySegment vSeg, long vOffBytes, float expPrev,
            float expCurr, int headDim) {
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

    private static void writeNormalizedAccumulator(MemorySegment out, long outIndex, float[] acc, float invL,
            int headDim) {
        int upperBound = FLOAT_SPECIES.loopBound(headDim);
        FloatVector invVec = FloatVector.broadcast(FLOAT_SPECIES, invL);
        int j = 0;
        for (; j < upperBound; j += FLOAT_SPECIES.length()) {
            FloatVector.fromArray(FLOAT_SPECIES, acc, j)
                    .mul(invVec)
                    .intoMemorySegment(out, (outIndex + j) * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; j < headDim; j++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, outIndex + j, acc[j] * invL);
        }
    }
}
