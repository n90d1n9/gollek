package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Standard Layer Normalization kernel.
 * Mean subtraction followed by variance scaling.
 */
public final class LayerNormKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final float EPS = 1e-5f;

    private LayerNormKernel() {}

    public static void execute(
            MemorySegment out,
            MemorySegment x,
            MemorySegment weight, // gamma
            MemorySegment bias,   // beta
            int hidden
    ) {
        // 1. Calculate Mean
        float sum = 0.0f;
        int i = 0;
        FloatVector vsum = FloatVector.zero(SPECIES);
        for (; i <= hidden - SPECIES.length(); i += SPECIES.length()) {
            vsum = vsum.add(FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder()));
        }
        sum = vsum.reduceLanes(VectorOperators.ADD);
        for (; i < hidden; i++) {
            sum += x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
        }
        float mean = sum / hidden;

        // 2. Calculate Variance
        float sumSq = 0.0f;
        i = 0;
        FloatVector vsumSq = FloatVector.zero(SPECIES);
        FloatVector vMean = FloatVector.broadcast(SPECIES, mean);
        for (; i <= hidden - SPECIES.length(); i += SPECIES.length()) {
            FloatVector vx = FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector vDiff = vx.sub(vMean);
            vsumSq = vsumSq.add(vDiff.mul(vDiff));
        }
        sumSq = vsumSq.reduceLanes(VectorOperators.ADD);
        for (; i < hidden; i++) {
            float diff = x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES) - mean;
            sumSq += diff * diff;
        }
        float variance = (sumSq / hidden) + EPS;
        float invStd = (float) (1.0 / Math.sqrt(variance));

        // 3. Normalize and Scale
        i = 0;
        FloatVector vInvStd = FloatVector.broadcast(SPECIES, invStd);
        for (; i <= hidden - SPECIES.length(); i += SPECIES.length()) {
            FloatVector vx = FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector vg = FloatVector.fromMemorySegment(SPECIES, weight, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector vb = (bias != null) ? 
                FloatVector.fromMemorySegment(SPECIES, bias, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder()) :
                FloatVector.zero(SPECIES);
                
            FloatVector vOut = vx.sub(vMean).mul(vInvStd).mul(vg).add(vb);
            vOut.intoMemorySegment(out, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < hidden; i++) {
            float val = x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            float g = weight.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            float b = (bias != null) ? bias.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES) : 0.0f;
            out.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, (val - mean) * invStd * g + b);
        }
    }
}
