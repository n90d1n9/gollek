package tech.kayys.gollek.ml.autograd;

import java.lang.foreign.MemorySegment;

/**
 * Abstraction for physical tensor data storage.
 * Allows tensors to live on the Java Heap (float[]) or in 
 * Unified Memory (MemorySegment) for zero-copy GPU access.
 */
public interface TensorStorage {
    
    /** Returns a float array view. May copy if data is not on heap. */
    float[] asArray();
    
    /** Returns an FFM MemorySegment view for native/GPU access. */
    MemorySegment asSegment();
    
    /** Returns the total number of elements. */
    int size();

    /** Creates a deep copy of this storage. */
    TensorStorage duplicate();

    /** Storage type. */
    enum Type { HEAP, UNIFIED }
    Type type();
}
