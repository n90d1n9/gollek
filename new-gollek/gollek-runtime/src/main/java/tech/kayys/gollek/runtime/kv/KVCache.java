package tech.kayys.gollek.runtime.kv;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class KVCache {
    private final CpuBuffer kBuf;
    private final CpuBuffer vBuf;
    private final KVCodec codec;
    private final int maxSeq;
    private final int heads;
    private final int headDim;
    private final int elemsPerToken;
    private final AtomicInteger position = new AtomicInteger(0);
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public KVCache(int maxSeq, int heads, int headDim, KVCodec codec) {
        this.codec = codec;
        this.maxSeq = maxSeq;
        this.heads = heads;
        this.headDim = headDim;
        this.elemsPerToken = heads * headDim;
        long totalElems = (long) maxSeq * elemsPerToken;
        long totalBytes = totalElems * codec.bytesPerElement();
        this.kBuf = new CpuBuffer(totalBytes);
        this.vBuf = new CpuBuffer(totalBytes);
    }

    public int position() {
        return position.get();
    }

    public int capacity() {
        return maxSeq;
    }

    public boolean canAppend() {
        return position.get() < maxSeq;
    }

    public void append(MemorySegment kNew, MemorySegment vNew) {
        lock.writeLock().lock();
        try {
            int pos = position.get();
            if (pos >= maxSeq) {
                throw new IllegalStateException(
                        String.format("KVCache overflow: position=%d, capacity=%d",
                                pos, maxSeq));
            }

            long offsetBytes = (long) pos * elemsPerToken * codec.bytesPerElement();
            MemorySegment kDst = kBuf.segment().asSlice(offsetBytes);
            MemorySegment vDst = vBuf.segment().asSlice(offsetBytes);

            codec.encode(kNew, kDst, elemsPerToken);
            codec.encode(vNew, vDst, elemsPerToken);

            position.incrementAndGet();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public void decodeK(MemorySegment dst) {
        lock.readLock().lock();
        try {
            int elems = position.get() * elemsPerToken;
            codec.decode(kBuf.segment(), dst, elems);
        } finally {
            lock.readLock().unlock();
        }
    }

    public void decodeV(MemorySegment dst) {
        lock.readLock().lock();
        try {
            int elems = position.get() * elemsPerToken;
            codec.decode(vBuf.segment(), dst, elems);
        } finally {
            lock.readLock().unlock();
        }
    }

    public MemorySegment rawK() {
        return kBuf.segment();
    }

    public MemorySegment rawV() {
        return vBuf.segment();
    }

    public int heads() {
        return heads;
    }

    public int headDim() {
        return headDim;
    }

    public KVCodec codec() {
        return codec;
    }

    // Reset for reuse
    public void clear() {
        lock.writeLock().lock();
        try {
            position.set(0);
        } finally {
            lock.writeLock().unlock();
        }
    }
}