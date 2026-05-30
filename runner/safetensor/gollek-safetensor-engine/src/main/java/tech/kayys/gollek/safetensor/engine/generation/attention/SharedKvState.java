/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.lang.foreign.MemorySegment;

public class SharedKvState implements AutoCloseable {
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

    @Override
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
