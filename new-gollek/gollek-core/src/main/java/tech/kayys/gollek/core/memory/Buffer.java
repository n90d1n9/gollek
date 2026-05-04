package tech.kayys.gollek.core.memory;

import java.lang.foreign.MemorySegment;

public interface Buffer {
    MemorySegment segment();

    long sizeBytes();

    void retain();

    void release();
}