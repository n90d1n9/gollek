package tech.kayys.gollek.inference.nativeimpl;

import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ExecutorService;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import tech.kayys.gollek.gguf.loader.TensorData;
import tech.kayys.gollek.gguf.loader.GGUFDequantizer;

/**
 * SIMD-optimized Logit Projection Kernel.
 * Computes: [1, hidden] x [hidden, vocab_size] -> [1, vocab_size]
 */
public final class LogitProjectionKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void execute(
        MemorySegment x,          // [hidden]
        TensorData outputWeight, // [vocab_size, hidden]
        MemorySegment logits,       // [vocab_size]
        int hidden,
        int vocabSize,
        float softCap,
        ExecutorService executor
    ) {
        // Parallelize across vocab chunks to avoid ParallelStream overhead
        int numTasks = Math.min(Runtime.getRuntime().availableProcessors() * 2, vocabSize / 256 + 1);
        int rowsPerTask = (vocabSize + numTasks - 1) / numTasks;

        List<Callable<Void>> tasks = new ArrayList<>();
        for (int t = 0; t < numTasks; t++) {
            final int start = t * rowsPerTask;
            final int end = Math.min(start + rowsPerTask, vocabSize);
            if (start >= end) break;

            tasks.add(() -> {
                // Pre-allocate dequantization buffer only if needed per task
                try (java.lang.foreign.Arena taskArena = java.lang.foreign.Arena.ofConfined()) {
                    MemorySegment rowF32 = null;
                    if (!outputWeight.isF32()) {
                        rowF32 = taskArena.allocate((long) hidden * Float.BYTES, 64);
                    }
                    
                    MemorySegment wBase = outputWeight.segment();
                    
                    for (int v = start; v < end; v++) {
                        float sum;
                        if (outputWeight.isQ8_0()) {
                            long bytesPerRow = (hidden / 32) * 34L;
                            GGUFDequantizer.dequantizeQ8_0(wBase, (long) v * bytesPerRow, rowF32, hidden);
                            sum = dot(x, rowF32, 0, hidden);
                        } else if (outputWeight.isF16()) {
                            GGUFDequantizer.dequantizeF16(wBase, (long) v * hidden * 2L, rowF32, hidden);
                            sum = dot(x, rowF32, 0, hidden);
                        } else {
                            // F32 - use absolute offset
                            sum = dot(x, wBase, (long) v * hidden * Float.BYTES, hidden);
                        }
                        
                        if (softCap > 0.0f) {
                            sum = (float) (softCap * Math.tanh(sum / softCap));
                        }
                        
                        logits.set(ValueLayout.JAVA_FLOAT, (long) v * Float.BYTES, sum);
                    }
                }
                return null;
            });
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Logit projection interrupted", e);
        }
    }

    private static float dot(MemorySegment x, MemorySegment w, long wOffset, int hidden) {
        int i = 0;
        int upperBound = SPECIES.loopBound(hidden);
        FloatVector acc = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector vx = FloatVector.fromMemorySegment(SPECIES, x, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector vw = FloatVector.fromMemorySegment(SPECIES, w, wOffset + (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            acc = vx.fma(vw, acc);
        }
        
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < hidden; i++) {
            sum += x.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES) * 
                   w.get(ValueLayout.JAVA_FLOAT, wOffset + (long) i * Float.BYTES);
        }
        return sum;
    }
}
