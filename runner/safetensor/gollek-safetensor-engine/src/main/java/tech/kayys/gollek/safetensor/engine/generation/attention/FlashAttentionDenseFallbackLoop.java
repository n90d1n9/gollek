/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

final class FlashAttentionDenseFallbackLoop {
    private FlashAttentionDenseFallbackLoop() {
    }

    static AccelTensor compute(AccelTensor q, KeyValueSource keyValueSource,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int slidingWindow = slidingWindow(config, layerIdx);

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
                AttentionOnlineSoftmax softmax = new AttentionOnlineSoftmax(new float[headDim], headDim);
                for (int i = 0; i < seqLenQ; i++) {
                    scanQueryPosition(qSeg, oSeg, keyValueSource, softmax,
                            ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2),
                            ((long) b * oStride0 + (long) i * oStride1 + (long) h * oStride2),
                            b, i, kvHeadIdx, startPos, slidingWindow, headDim, scale, causal, softCap);
                }
            }
        }

        return out;
    }

    private static void scanQueryPosition(MemorySegment qSeg, MemorySegment outSeg, KeyValueSource keyValueSource,
            AttentionOnlineSoftmax softmax, long qOffset, long outOffset, int batch, int queryIndex, int kvHeadIdx,
            int startPos, int slidingWindow, int headDim, float scale, boolean causal, float softCap) {
        softmax.reset();
        int minPos = slidingWindow == Integer.MAX_VALUE
                ? 0
                : Math.max(0, startPos + queryIndex - slidingWindow + 1);

        for (int token = 0; token < keyValueSource.totalTokens(); token++) {
            if (token < minPos) {
                continue;
            }
            if (causal && token > startPos + queryIndex) {
                break;
            }

            float score = AttentionFp32VectorMath.dotProduct(
                    qSeg, qOffset * Float.BYTES,
                    keyValueSource.keySegment(), keyValueSource.keyOffset(batch, token, kvHeadIdx) * Float.BYTES,
                    headDim) * scale;
            if (softCap > 0.0f) {
                score = (float) (Math.tanh(score / softCap) * softCap);
            }

            softmax.observe(score);
            AttentionFp32VectorMath.updateAccumulator(
                    softmax.accumulator(),
                    keyValueSource.valueSegment(),
                    keyValueSource.valueOffset(batch, token, kvHeadIdx) * Float.BYTES,
                    softmax.previousWeight(),
                    softmax.currentWeight(),
                    headDim);
        }

        AttentionFp32VectorMath.writeNormalizedAccumulator(
                outSeg, outOffset, softmax.accumulator(), softmax.inverseNormalizer(), headDim);
    }

    private static int slidingWindow(ModelConfig config, int layerIdx) {
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        return slidingLayer ? config.slidingWindowSize() : Integer.MAX_VALUE;
    }

    interface KeyValueSource {
        int totalTokens();

        MemorySegment keySegment();

        MemorySegment valueSegment();

        long keyOffset(int batch, int token, int kvHeadIdx);

        long valueOffset(int batch, int token, int kvHeadIdx);
    }
}
