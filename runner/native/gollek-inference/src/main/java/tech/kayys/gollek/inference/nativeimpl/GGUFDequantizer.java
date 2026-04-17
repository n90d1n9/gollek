package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Utility to dequantize GGUF/GGML tensors into F32.
 */
public final class GGUFDequantizer {

    public static void dequantizeQ8_0(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / 32;
        for (long b = 0; b < numBlocks; b++) {
            long srcOffset = b * 34; // 2 (half) + 32 (int8)
            short deltaBits = src.get(ValueLayout.JAVA_SHORT, srcOffset);
            float delta = halfToFloat(deltaBits);
            
            for (int i = 0; i < 32; i++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, srcOffset + 2 + i);
                dst.set(ValueLayout.JAVA_FLOAT, (b * 32 + i) * Float.BYTES, q * delta);
            }
        }
    }

    public static void dequantizeF16(MemorySegment src, MemorySegment dst, long numElements) {
        for (long i = 0; i < numElements; i++) {
            short bits = src.get(ValueLayout.JAVA_SHORT, i * 2);
            dst.set(ValueLayout.JAVA_FLOAT, i * Float.BYTES, halfToFloat(bits));
        }
    }

    private static float halfToFloat(short h) {
        return Float.float16ToFloat(h);
    }
}
