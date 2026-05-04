package tech.kayys.gollek.core.memory;

import java.lang.foreign.*;

public final class CpuBuffer implements Buffer {
    private final Arena arena;
    private final MemorySegment segment;
    private int refCount = 1;

    public CpuBuffer(long sizeBytes) {
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(sizeBytes);
    }

    @Override
    public MemorySegment segment() {
        return segment;
    }

    @Override
    public long sizeBytes() {
        return segment.byteSize();
    }

    @Override
    public synchronized void retain() {
        refCount++;
    }

    @Override
    public synchronized void release() {
        refCount--;
        if (refCount == 0) {
            arena.close();
        }
    }
}