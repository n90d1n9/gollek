/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class FlashAttentionJavaFallback {
    private FlashAttentionJavaFallback() {
    }

    static AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            ModelConfig config, int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        FlashAttentionDenseFallbackLoop.KeyValueSource source = new StridedKeyValueSource(
                k.dataSegment(),
                v.dataSegment(),
                Math.toIntExact(k.size(1)),
                k.stride()[0],
                k.stride()[1],
                k.stride()[2],
                v.stride()[0],
                v.stride()[1],
                v.stride()[2]);
        return FlashAttentionDenseFallbackLoop.compute(
                q, source, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, causal, softCap);
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
        FlashAttentionDenseFallbackLoop.KeyValueSource source = new GatheredKeyValueSource(
                kSeg, vSeg, startPos + Math.toIntExact(q.size(1)), numKVHeads, headDim);
        return FlashAttentionDenseFallbackLoop.compute(
                q, source, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, causal, softCap);
    }

    private record StridedKeyValueSource(
            MemorySegment keySegment,
            MemorySegment valueSegment,
            int totalTokens,
            long keyBatchStride,
            long keyTokenStride,
            long keyHeadStride,
            long valueBatchStride,
            long valueTokenStride,
            long valueHeadStride) implements FlashAttentionDenseFallbackLoop.KeyValueSource {

        @Override
        public long keyOffset(int batch, int token, int kvHeadIdx) {
            return (long) batch * keyBatchStride + (long) token * keyTokenStride + (long) kvHeadIdx * keyHeadStride;
        }

        @Override
        public long valueOffset(int batch, int token, int kvHeadIdx) {
            return (long) batch * valueBatchStride + (long) token * valueTokenStride
                    + (long) kvHeadIdx * valueHeadStride;
        }
    }

    private record GatheredKeyValueSource(
            MemorySegment keySegment,
            MemorySegment valueSegment,
            int totalTokens,
            int numKVHeads,
            int headDim) implements FlashAttentionDenseFallbackLoop.KeyValueSource {

        @Override
        public long keyOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }

        @Override
        public long valueOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }
    }
}
