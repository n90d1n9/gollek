package tech.kayys.gollek.ml.autograd;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Standard JVM Heap storage using float[].
 */
public class HeapStorage implements TensorStorage {
    private final float[] data;

    public HeapStorage(float[] data) {
        this.data = Objects.requireNonNull(data);
    }

    @Override
    public float[] asArray() {
        return data;
    }

    @Override
    public MemorySegment asSegment() {
        // Wraps the heap array into a MemorySegment zero-copy
        return MemorySegment.ofArray(data);
    }

    @Override
    public int size() {
        return data.length;
    }

    @Override
    public TensorStorage duplicate() {
        return new HeapStorage(data.clone());
    }

    @Override
    public Type type() {
        return Type.HEAP;
    }
}
