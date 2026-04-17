package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated RoPE (Rotary Positional Embedding) Kernel.
 */
public final class RoPEKernel {

    private RoPEKernel() {}

    public static void apply(
            MemorySegment qOrK, // [head_dim]
            int offset,         // position index
            RoPECache table,
            int dim,
            boolean isNeox
    ) {
        int half = dim / 2;
        MemorySegment cos = table.cos;
        MemorySegment sin = table.sin;

        if (isNeox) {
            // Neox style: x[i] pairs with x[i + half]
            for (int i = 0; i < half; i++) {
                long idx = ((long) offset * half + i) * 4L;

                float c = cos.get(ValueLayout.JAVA_FLOAT, idx);
                float s = sin.get(ValueLayout.JAVA_FLOAT, idx);

                float x0 = qOrK.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
                float x1 = qOrK.get(ValueLayout.JAVA_FLOAT, (long) (i + half) * 4L);

                float y0 = x0 * c - x1 * s;
                float y1 = x0 * s + x1 * c;

                qOrK.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, y0);
                qOrK.set(ValueLayout.JAVA_FLOAT, (long) (i + half) * 4L, y1);
            }
        } else {
            // LLaMA style (GPT-J): x[i*2] pairs with x[i*2 + 1]
            for (int i = 0; i < half; i++) {
                long idx = ((long) offset * half + i) * 4L;

                float c = cos.get(ValueLayout.JAVA_FLOAT, idx);
                float s = sin.get(ValueLayout.JAVA_FLOAT, idx);

                float x0 = qOrK.get(ValueLayout.JAVA_FLOAT, (long) (i * 2) * 4L);
                float x1 = qOrK.get(ValueLayout.JAVA_FLOAT, (i * 2L + 1) * 4L);

                float y0 = x0 * c - x1 * s;
                float y1 = x0 * s + x1 * c;

                qOrK.set(ValueLayout.JAVA_FLOAT, (long) (i * 2) * 4L, y0);
                qOrK.set(ValueLayout.JAVA_FLOAT, (i * 2L + 1) * 4L, y1);
            }
        }
    }
}
