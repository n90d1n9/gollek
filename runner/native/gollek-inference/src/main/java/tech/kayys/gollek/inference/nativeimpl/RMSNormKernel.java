package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated RMSNorm (Root Mean Square Normalization).
 */
public final class RMSNormKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private RMSNormKernel() {}

    public static void execute(
            MemorySegment input,
            MemorySegment output,
            MemorySegment weight,
            int size,
            float eps
    ) {
        int i = 0;

        FloatVector vsum = FloatVector.zero(SPECIES);

        // 1. compute sum(x^2)
        for (; i <= size - SPECIES.length(); i += SPECIES.length()) {
            var vx = FloatVector.fromMemorySegment(
                    SPECIES, input, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());

            vsum = vx.fma(vx, vsum); // vx * vx + vsum
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        // tail
        for (; i < size; i++) {
            float x = input.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            sum += x * x;
        }

        float mean = sum / size;
        float invRms = (float) (1.0 / Math.sqrt(mean + eps));

        // 2. normalize + scale
        i = 0;
        FloatVector vInv = FloatVector.broadcast(SPECIES, invRms);

        for (; i <= size - SPECIES.length(); i += SPECIES.length()) {
            var vx = FloatVector.fromMemorySegment(
                    SPECIES, input, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());

            var vw = FloatVector.fromMemorySegment(
                    SPECIES, weight, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());

            var vy = vx.mul(vInv).mul(vw);

            vy.intoMemorySegment(output, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }

        // tail
        for (; i < size; i++) {
            float x = input.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            float w = weight.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            float y = x * invRms * w;

            output.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, y);
        }
    }
}
