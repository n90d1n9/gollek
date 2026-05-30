/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

final class PagedKvCacheIO {
    private PagedKvCacheIO() {
    }

    static void updateCache(AccelTensor key, AccelTensor value, KVCacheManager.KVCacheSession kvSession, int layerIdx,
            int startPos, int seqLen, int numHeads, int headDim) {
        BlockManager blockManager = kvSession.blockManager();
        MemorySegment keySeg = key.dataPtr();
        MemorySegment valueSeg = value.dataPtr();

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

            MemorySegment keyBlock = blockManager.getKBlock(blockIdx);
            MemorySegment valueBlock = blockManager.getVBlock(blockIdx);
            MemorySegment keyScaleBlock = blockManager.getKScaleBlock(blockIdx);
            MemorySegment valueScaleBlock = blockManager.getVScaleBlock(blockIdx);

            for (int h = 0; h < numHeads; h++) {
                long srcOff = ((long) s * numHeads + h) * headDim;
                long hDstOff = ((long) h * headStride + (long) tokenIdxInBlock * tokenStride);

                if (quantizedInt8) {
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokenIdxInBlock;
                    quantizeVectorToInt8(keySeg, srcOff, keyBlock, hDstOff, keyScaleBlock, scaleIndex, headDim);
                    quantizeVectorToInt8(valueSeg, srcOff, valueBlock, hDstOff, valueScaleBlock, scaleIndex, headDim);
                } else if (quantizedInt4) {
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokenIdxInBlock;
                    quantizeVectorToInt4(keySeg, srcOff, keyBlock, hDstOff, keyScaleBlock, scaleIndex, headDim);
                    quantizeVectorToInt4(valueSeg, srcOff, valueBlock, hDstOff, valueScaleBlock, scaleIndex, headDim);
                } else {
                    MemorySegment.copy(keySeg, ValueLayout.JAVA_FLOAT, srcOff * 4, keyBlock, ValueLayout.JAVA_FLOAT,
                            hDstOff * 4, headDim);
                    MemorySegment.copy(valueSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, valueBlock, ValueLayout.JAVA_FLOAT,
                            hDstOff * 4, headDim);
                }
            }
        }
    }

    static void gather(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession, int kvLayerIdx,
            int totalTokens, int numKVHeads, int headDim, MemorySegment keyOut, MemorySegment valueOut) {
        List<Integer> blocks = kvSession.getBlockIndices(kvLayerIdx);
        int blockSize = kvSession.tokensPerBlock();
        BlockManager.KvStorageType storageType = blockManager.getStorageType();
        for (int blk = 0; blk < blocks.size(); blk++) {
            int phys = blocks.get(blk);
            int tokenStart = blk * blockSize;
            int tokenEnd = Math.min(totalTokens, tokenStart + blockSize);
            MemorySegment keyBlock = blockManager.getKBlock(phys);
            MemorySegment valueBlock = blockManager.getVBlock(phys);
            MemorySegment keyScaleBlock = blockManager.getKScaleBlock(phys);
            MemorySegment valueScaleBlock = blockManager.getVScaleBlock(phys);

            for (int tok = tokenStart; tok < tokenEnd; tok++) {
                int tokInBlock = tok - tokenStart;
                for (int h = 0; h < numKVHeads; h++) {
                    long srcElement = ((long) h * blockManager.getHeadStride())
                            + ((long) tokInBlock * blockManager.getTokenStride());
                    long dstElement = ((long) tok * numKVHeads + h) * headDim;
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tokInBlock;
                    writeVectorAsFloat(storageType, keyBlock, keyScaleBlock, srcElement, scaleIndex, keyOut,
                            dstElement, headDim);
                    writeVectorAsFloat(storageType, valueBlock, valueScaleBlock, srcElement, scaleIndex, valueOut,
                            dstElement, headDim);
                }
            }
        }
    }

    static void packDenseSharedKvIntoTemporaryPagedPool(AccelTensor key, AccelTensor value,
            int numKVHeads, int headDim, MemorySegment packedKey, MemorySegment packedValue) {
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
                    MemorySegment.copy(keySeg, srcKElement * Float.BYTES, packedKey, dstElement * Float.BYTES, bytes);
                    MemorySegment.copy(valueSeg, srcVElement * Float.BYTES, packedValue, dstElement * Float.BYTES,
                            bytes);
                }
            }
        }
    }

    static MaterializedKvPools materializeBlocksForMetal(BlockManager blockManager,
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
                    long srcElement = ((long) h * blockManager.getHeadStride())
                            + ((long) tok * blockManager.getTokenStride());
                    long scaleIndex = (long) h * blockManager.getScaleStride() + tok;
                    long dstElement = ((long) localBlockIdx * blockElements)
                            + ((long) h * blockSize * headDim)
                            + ((long) tok * headDim);
                    writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock, srcElement, scaleIndex, packedK,
                            dstElement, headDim);
                    writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock, srcElement, scaleIndex, packedV,
                            dstElement, headDim);
                }
            }
        }

        return new MaterializedKvPools(packedK, packedV);
    }

    static void packRangeIntoTemporaryPagedPool(BlockManager blockManager, KVCacheManager.KVCacheSession kvSession,
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
                    writeVectorAsFloat(storageType, srcKBlock, srcKScaleBlock, srcElement, scaleIndex, packedK,
                            dstElement, headDim);
                    writeVectorAsFloat(storageType, srcVBlock, srcVScaleBlock, srcElement, scaleIndex, packedV,
                            dstElement, headDim);
                }
            }
        }
    }

    private static void quantizeVectorToInt8(MemorySegment srcSeg, long srcFloatIndex,
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

    private static void quantizeVectorToInt4(MemorySegment srcSeg, long srcFloatIndex,
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

    private static void writeVectorAsFloat(BlockManager.KvStorageType storageType,
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

    private static void writePackedSignedInt4(MemorySegment dstSeg, long dstElementIndex, int quantized) {
        long byteIndex = dstElementIndex >>> 1;
        int stored = quantized + 8;
        int existing = Byte.toUnsignedInt(dstSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int packed = (dstElementIndex & 1L) == 0L
                ? ((existing & 0xF0) | (stored & 0x0F))
                : ((existing & 0x0F) | ((stored & 0x0F) << 4));
        dstSeg.setAtIndex(ValueLayout.JAVA_BYTE, byteIndex, (byte) packed);
    }

    private static int readPackedSignedInt4(MemorySegment srcSeg, long srcElementIndex) {
        long byteIndex = srcElementIndex >>> 1;
        int packed = Byte.toUnsignedInt(srcSeg.getAtIndex(ValueLayout.JAVA_BYTE, byteIndex));
        int nibble = (srcElementIndex & 1L) == 0L ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return nibble - 8;
    }

    record MaterializedKvPools(MemorySegment kPool, MemorySegment vPool) {
    }
}
