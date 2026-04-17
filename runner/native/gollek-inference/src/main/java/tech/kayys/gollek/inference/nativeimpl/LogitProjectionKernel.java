package tech.kayys.gollek.inference.nativeimpl;

import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * SIMD-optimized Logit Projection Kernel.
 * Computes: [1, hidden] x [hidden, vocab_size] -> [1, vocab_size]
 * <p>
 * This is effectively a large batch of dot products.
 */
public final class LogitProjectionKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void execute(
        MemorySegment x,          // [hidden]
        MemorySegment outputWeight, // [vocab_size, hidden]
        MemorySegment logits,       // [vocab_size]
        int hidden,
        int vocabSize
    ) {
        // Paraellize across tokens
        java.util.stream.IntStream.range(0, vocabSize).parallel().forEach(v -> {
            float sum = 0.0f;
            long weightOffset = (long) v * hidden * Float.BYTES;
            
            int i = 0;
            int upperBound = SPECIES.loopBound(hidden);
            
            FloatVector acc = FloatVector.zero(SPECIES);
            
            for (; i < upperBound; i += SPECIES.length()) {
                FloatVector vx = FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                FloatVector vw = FloatVector.fromMemorySegment(SPECIES, outputWeight, weightOffset + (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
                acc = vx.fma(vw, acc);
            }
            
            sum = acc.reduceLanes(VectorOperators.ADD);
            
            // Tail loop
            for (; i < hidden; i++) {
                sum += x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES) * 
                       outputWeight.get(ValueLayout.JAVA_FLOAT, weightOffset + (long) i * Float.BYTES);
            }
            
            logits.set(ValueLayout.JAVA_FLOAT, (long) v * Float.BYTES, sum);
        });
    }
}
