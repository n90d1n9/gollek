package tech.kayys.gollek.ml.autograd;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * High-performance Unified Memory storage using FFM MemorySegment.
 * Highly optimized for Apple Silicon (Mac M-Series) Metal backend.
 * Provides zero-copy access to the GPU at the virtual address level.
 */
public class UnifiedStorage implements TensorStorage {
    
    // We use a GC-managed 'auto' arena so tensor memory is freed 
    // when the GradTensor object is garbage collected.
    private final MemorySegment segment;
    private final int numel;

    public UnifiedStorage(int numel) {
        this.numel = numel;
        // Allocate 64-byte aligned memory for optimal MPS performance
        this.segment = Arena.ofAuto().allocate((long) numel * 4L, 64L);
    }

    public UnifiedStorage(MemorySegment segment, int numel) {
        this.segment = Objects.requireNonNull(segment);
        this.numel = numel;
    }

    @Override
    public float[] asArray() {
        // Warning: This involves a heap-to-native copy. 
        // We should avoid calling this in 'Hot' paths on Metal.
        float[] arr = new float[numel];
        MemorySegment.copy(segment, ValueLayout.JAVA_FLOAT, 0, arr, 0, numel);
        return arr;
    }

    @Override
    public MemorySegment asSegment() {
        return segment;
    }

    @Override
    public int size() {
        return numel;
    }

    @Override
    public TensorStorage duplicate() {
        UnifiedStorage copy = new UnifiedStorage(numel);
        copy.asSegment().copyFrom(this.segment);
        return copy;
    }

    @Override
    public Type type() {
        return Type.UNIFIED;
    }
}
