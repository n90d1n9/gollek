package tech.kayys.gollek.runtime.inference.kv;

import java.lang.foreign.MemorySegment;

/**
 * Local TurboQuant types for runtime-inference module.
 * Avoids cyclic dependency with gollek-quantizer-turboquant.
 */
public final class TurboQuantTypes {

    private TurboQuantTypes() {}

    /**
     * Result of TurboQuant_prod quantization.
     */
    public record QuantProdResult(
        int[] mseIndices,
        byte[] qjlSigns,
        float residualNorm
    ) {}

    /**
     * TurboQuant configuration.
     */
    public record TurboQuantConfig(
        int bits,
        int dimension,
        long seed
    ) {
        public static TurboQuantConfig prod3bitKvCache(int dimension) {
            return new TurboQuantConfig(3, dimension, 42L);
        }

        public int mseStageBits() {
            return bits - 1;
        }
    }

    /**
     * FFM bulk copy utilities.
     */
    public static void bulkCopyToNative(float[] src, MemorySegment dest, long offsetBytes) {
        long bytes = (long) src.length * Float.BYTES;
        MemorySegment srcSegment = MemorySegment.ofArray(src);
        dest.asSlice(offsetBytes, bytes).copyFrom(srcSegment);
    }

    public static void bulkCopyFromNative(MemorySegment src, long offsetBytes, float[] dest) {
        long bytes = (long) dest.length * Float.BYTES;
        MemorySegment destSegment = MemorySegment.ofArray(dest);
        destSegment.copyFrom(src.asSlice(offsetBytes, bytes));
    }

    public static void bulkCopyByteToNative(byte[] src, MemorySegment dest, long offsetBytes) {
        MemorySegment srcSegment = MemorySegment.ofArray(src);
        dest.asSlice(offsetBytes, src.length).copyFrom(srcSegment);
    }

    public static void bulkCopyFromNativeByte(MemorySegment src, long offsetBytes, byte[] dest) {
        MemorySegment destSegment = MemorySegment.ofArray(dest);
        destSegment.copyFrom(src.asSlice(offsetBytes, dest.length));
    }
}
