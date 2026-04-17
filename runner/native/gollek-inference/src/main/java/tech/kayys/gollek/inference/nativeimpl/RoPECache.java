package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Precomputed RoPE (Rotary Positional Embedding) table.
 * Caches sine and cosine values to avoid expensive trigonometric functions during the forward pass.
 */
public final class RoPECache {

    public final MemorySegment cos;
    public final MemorySegment sin;

    public final int maxSeq;
    public final int dim;

    private RoPECache(MemorySegment cos, MemorySegment sin, int maxSeq, int dim) {
        this.cos = cos;
        this.sin = sin;
        this.maxSeq = maxSeq;
        this.dim = dim;
    }

    public static RoPECache build(Arena arena, int maxSeq, int dim, float base) {
        int half = dim / 2;
        long size = (long) maxSeq * half * 4;

        MemorySegment cosSeg = arena.allocate(size, 64);
        MemorySegment sinSeg = arena.allocate(size, 64);

        for (int pos = 0; pos < maxSeq; pos++) {
            for (int i = 0; i < half; i++) {
                double theta = pos / Math.pow(base, (2.0 * i) / dim);

                float c = (float) Math.cos(theta);
                float s = (float) Math.sin(theta);

                long idx = ((long) pos * half + i) * 4;

                cosSeg.set(ValueLayout.JAVA_FLOAT, idx, c);
                sinSeg.set(ValueLayout.JAVA_FLOAT, idx, s);
            }
        }

        return new RoPECache(cosSeg, sinSeg, maxSeq, dim);
    }
}
