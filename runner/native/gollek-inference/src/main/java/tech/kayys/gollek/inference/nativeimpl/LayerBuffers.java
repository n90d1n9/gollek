package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.Arena;

/**
 * Reusable runtime buffers required for a single transformer layer tick.
 * Eliminates per-token memory allocation.
 */
public final class LayerBuffers {

    public final MemorySegment norm;     // [hidden]
    public final MemorySegment q;        // [heads * headDim]
    public final MemorySegment k;
    public final MemorySegment v;
    public final MemorySegment attnOut;  // [hidden]
    public final MemorySegment ffn;      // [ffnDim]

    public LayerBuffers(Arena arena, int hidden, int nHeadsKv, int headDim, int ffnDim) {
        // Allocate with 64-byte alignment for SIMD
        norm = arena.allocate(hidden * 4L, 64);
        q = arena.allocate(hidden * 4L, 64);
        k = arena.allocate((long) nHeadsKv * headDim * 4L, 64);
        v = arena.allocate((long) nHeadsKv * headDim * 4L, 64);
        attnOut = arena.allocate(hidden * 4L, 64);
        ffn = arena.allocate(ffnDim * 4L, 64);
    }
}
