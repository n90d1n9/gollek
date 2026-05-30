/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionMetalAttention {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<MetalFlashAttentionBinding> metalFa4;
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;

    FlashAttentionMetalAttention(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy) {
        this.canUseMetal = canUseMetal;
        this.metalBinding = metalBinding;
        this.metalFa4 = metalFa4;
        this.routingPolicy = routingPolicy;
    }

    AccelTensor denseSharedAttention(AccelTensor q, AccelTensor k, AccelTensor v,
            SharedKvState sharedKvState, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap) {
        if (!canUseMetal.getAsBoolean()) {
            return null;
        }
        MetalBinding binding = metalBinding();
        if (binding == null || !binding.isRuntimeActive()) {
            return null;
        }
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = Math.toIntExact(k.size(1));
        if (batch <= 0 || seqLenQ <= 0 || totalTokens <= 0) {
            return null;
        }

        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        if (slidingLayer && !binding.isWindowedAttentionAvailable()) {
            return null;
        }

        FlashAttentionRoutingPolicy routing = routingPolicy();
        boolean usePackedSharedDecode = routing.shouldUsePackedSharedDecodeAttention(config, modelPolicy, seqLenQ,
                sharedKvState);
        boolean useFa4 = routing.canUseFa4Attention(softCap) && !usePackedSharedDecode;
        if (!useFa4 && !usePackedSharedDecode && !routing.allowLegacyMetalAttentionBridge(modelPolicy)) {
            return null;
        }
        AccelTensor qContiguous = q.contiguous();
        AccelTensor kContiguous = null;
        AccelTensor vContiguous = null;
        try (Arena arena = Arena.ofConfined()) {
            AccelTensor out = AccelTensor.zeros(q.shape());

            if (useFa4) {
                MetalFlashAttentionBinding fa4 = metalFa4();
                if (fa4 == null || !fa4.isNativeAvailable()) {
                    out.close();
                    return null;
                }
                kContiguous = k.contiguous();
                vContiguous = v.contiguous();
                boolean useBf16 = fa4.isBf16Available()
                        && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = fa4.fa4Attention(
                        out.dataPtr(), qContiguous.dataPtr(), kContiguous.dataPtr(), vContiguous.dataPtr(),
                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), totalTokens, numQHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath("dense_shared_fa4");
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
                    PagedKvCacheIO.packDenseSharedKvIntoTemporaryPagedPool(kContiguous, vContiguous, numKVHeads,
                            headDim, packedK, packedV);
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
                                ? binding.attentionWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap)
                                : binding.attentionGqaWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap))
                        : (numKVHeads == numQHeads
                                ? binding.attention(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap)
                                : binding.attentionGqa(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap));
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath(usePackedSharedDecode
                            ? "dense_shared_packed_metal"
                            : "dense_shared_legacy_metal");
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

    AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int startPos, int numHeads, int numKVHeads, int headDim, float scale, boolean causal, float softCap,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        FlashAttentionRoutingPolicy routing = routingPolicy();
        boolean canUseFa4Path = routing.canUseFa4Attention(softCap);
        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        boolean preferPagedMetalFirst = routing.preferPagedMetalAttentionBeforeFa4(modelPolicy, (int) seqLen,
                totalTokens);

        if (preferPagedMetalFirst && !kvSession.isQuantized()) {
            AccelTensor pagedOut = tryPagedMetalAttention(q, kvSession, blockManager, blocks, kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens, batch,
                    seqLen, slidingLayer, null, modelPolicy);
            if (pagedOut != null) {
                DirectInferenceProfiler.recordAttentionPath(
                        routing.pagedAttentionPathName(slidingLayer, seqLen, totalTokens, true));
                return pagedOut;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            AccelTensor out = AccelTensor.zeros(q.shape());
            MetalFlashAttentionBinding fa4 = metalFa4();
            if (!preferPagedMetalFirst && canUseFa4Path && fa4 != null && fa4.isNativeAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                PagedKvCacheIO.gather(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim,
                        kGathered, vGathered);

                boolean useBf16 = fa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = fa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath("fa4_gathered");
                    return out;
                }
                out.close();
                out = AccelTensor.zeros(q.shape());
            }
            AccelTensor pagedOut = tryPagedMetalAttention(q, kvSession, blockManager, blocks, kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, causal, softCap, config, layerIdx, totalTokens, batch,
                    seqLen, slidingLayer, arena, modelPolicy);
            if (pagedOut != null) {
                out.close();
                DirectInferenceProfiler.recordAttentionPath(
                        routing.pagedAttentionPathName(slidingLayer, seqLen, totalTokens, false));
                return pagedOut;
            }
            if (canUseFa4Path && fa4 != null && fa4.isNativeAvailable()) {
                long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
                MemorySegment kGathered = arena.allocate(gatherBytes, 64);
                MemorySegment vGathered = arena.allocate(gatherBytes, 64);
                PagedKvCacheIO.gather(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim,
                        kGathered, vGathered);

                boolean useBf16 = fa4.isBf16Available() && Boolean.getBoolean("gollek.safetensor.use_bf16_attention");
                int result = fa4.fa4Attention(
                        out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                        (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath("fa4_gathered_after_paged");
                    return out;
                }
                out.close();
            }
            DirectInferenceProfiler.recordAttentionPath("paged_java");
            return PagedAttentionVectorAPI.compute(q, null, kvSession, kvLayerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, causal, softCap);
        }
    }

    AccelTensor slidingDecodeAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            int layerIdx, int kvLayerIdx, int startPos, int numHeads, int numKVHeads, int headDim, float scale,
            float softCap, ModelConfig config, FlashAttentionModelPolicy modelPolicy) {
        FlashAttentionRoutingPolicy routing = routingPolicy();
        if (!routing.allowSlidingDecodeMetalAttentionBridge(config, modelPolicy, layerIdx, (int) q.size(1))) {
            DirectInferenceProfiler.recordAttentionPath("sliding_decode_java");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
        MetalBinding binding = metalBinding();
        if (binding == null || !binding.isRuntimeActive()) {
            DirectInferenceProfiler.recordAttentionPath("sliding_decode_java");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
        long batch = q.size(0);
        int seqLen = (int) q.size(1);
        int totalTokens = startPos + seqLen;
        int slidingWindow = config.slidingWindowSize();
        int contextStart = Math.max(0, totalTokens - slidingWindow);
        int contextTokens = totalTokens - contextStart;
        if (routing.shortDecodeUsesNativeAttention(contextTokens)) {
            DirectInferenceProfiler.recordAttentionPath("sliding_decode_native_short");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
        int blockSize = kvSession.tokensPerBlock();
        int maxBlocks = Math.max(1, (contextTokens + blockSize - 1) / blockSize);
        int poolElements = maxBlocks * numKVHeads * blockSize * headDim;
        int batchInt = Math.toIntExact(batch);

        if (routing.enableRawPagedSlidingDecodeAttention()) {
            AccelTensor rawPagedOut = tryPagedMetalAttention(q, kvSession, kvSession.blockManager(),
                    kvSession.getBlockIndices(kvLayerIdx), kvLayerIdx, startPos,
                    numHeads, numKVHeads, headDim, scale, true, softCap, config, layerIdx,
                    totalTokens, batch, seqLen, true, null, modelPolicy);
            if (rawPagedOut != null) {
                DirectInferenceProfiler.recordAttentionPath("sliding_decode_paged_metal");
                return rawPagedOut;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment packedK = arena.allocate((long) poolElements * Float.BYTES, 64);
            MemorySegment packedV = arena.allocate((long) poolElements * Float.BYTES, 64);
            PagedKvCacheIO.packRangeIntoTemporaryPagedPool(kvSession.blockManager(), kvSession, kvLayerIdx,
                    contextStart, totalTokens, numKVHeads, headDim, blockSize, packedK, packedV);

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
                    ? binding.attention(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap)
                    : binding.attentionGqa(
                            out.dataPtr(), q.dataPtr(), packedK, packedV,
                            blockTable, contextLens,
                            (int) batch, seqLen, numHeads, numKVHeads, headDim,
                            blockSize, maxBlocks,
                            scale, 0, softCap);
            if (result == 0) {
                DirectInferenceProfiler.recordAttentionPath("sliding_decode_metal");
                return out;
            }
            out.close();
            DirectInferenceProfiler.recordAttentionPath("sliding_decode_java");
            return PagedAttentionVectorAPI.compute(q, config, kvSession, layerIdx, kvLayerIdx, startPos, numHeads,
                    numKVHeads, headDim, scale, true, softCap);
        }
    }

    private AccelTensor tryPagedMetalAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            BlockManager blockManager, List<Integer> blocks, int kvLayerIdx, int startPos, int numHeads,
            int numKVHeads, int headDim, float scale, boolean causal, float softCap, ModelConfig config, int layerIdx,
            int totalTokens, long batch, long seqLen, boolean slidingLayer, Arena arena,
            FlashAttentionModelPolicy modelPolicy) {
        MetalBinding binding = metalBinding();
        if (!routingPolicy().allowPagedMetalAttentionBridge(modelPolicy, (int) seqLen, totalTokens)
                || binding == null
                || !binding.isRuntimeActive()
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
                PagedKvCacheIO.MaterializedKvPools materialized = PagedKvCacheIO.materializeBlocksForMetal(
                        blockManager, kvSession, blocks, numKVHeads, headDim, arena);
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
                            ? binding.attentionWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap)
                            : binding.attentionGqaWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap))
                    : (numKVHeads == numHeads
                            ? binding.attention(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, softCap)
                            : binding.attentionGqa(
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

    private MetalBinding metalBinding() {
        return metalBinding.get();
    }

    private MetalFlashAttentionBinding metalFa4() {
        return metalFa4.get();
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        return routingPolicy.get();
    }
}
