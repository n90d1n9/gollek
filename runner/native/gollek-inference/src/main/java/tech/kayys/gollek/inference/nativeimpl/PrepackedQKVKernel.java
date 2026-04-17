package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Optimized QKV Kernel using prepacked weights.
 * Performance optimized: avoids task allocation overhead for small head counts.
 */
public final class PrepackedQKVKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private PrepackedQKVKernel() {}

    public static void computeParallel(
            MemorySegment x,              // [hidden]
            MemorySegment wPacked,        // [numHeads + 2*numHeadsKv][headDim][hidden]
            MemorySegment bqkv,           // [numHeads + 2*numHeadsKv][headDim]
            MemorySegment outQ,
            MemorySegment outK,
            MemorySegment outV,
            int hidden,
            int numHeads,
            int numHeadsKv,
            int headDim,
            ExecutorService executor
    ) {
        // Parallelize by groups of heads to reduce task overhead
        int numTasks = Math.min(Runtime.getRuntime().availableProcessors(), numHeads);
        int headsPerTask = (numHeads + numTasks - 1) / numTasks;
        
        List<Callable<Void>> tasks = new ArrayList<>();
        for (int i = 0; i < numTasks; i++) {
            final int startHead = i * headsPerTask;
            final int endHead = Math.min(startHead + headsPerTask, numHeads);
            if (startHead >= endHead) break;

            tasks.add(() -> {
                for (int h = startHead; h < endHead; h++) {
                    computeHead(h, numHeads, numHeadsKv, headDim, hidden, x, wPacked, bqkv, outQ, outK, outV);
                }
                return null;
            });
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("QKV projection interrupted", e);
        }
    }

    private static void computeHead(int head, int numHeads, int numHeadsKv, int headDim, int hidden,
                                  MemorySegment x, MemorySegment wPacked, MemorySegment bqkv,
                                  MemorySegment outQ, MemorySegment outK, MemorySegment outV) {
        // Compute Q
        long qWeightBase = (long) head * headDim * hidden;
        long qBiasBase = (long) head * headDim;
        for (int d = 0; d < headDim; d++) {
            float sumQ = dot(x, wPacked, qWeightBase + (long) d * hidden, hidden);
            if (bqkv != null) sumQ += bqkv.get(ValueLayout.JAVA_FLOAT, (qBiasBase + d) * 4L);
            outQ.set(ValueLayout.JAVA_FLOAT, (qBiasBase + d) * 4L, sumQ);
        }

        // Compute K and V (only for heads that map to KV heads in GQA)
        if (head < numHeadsKv) {
            long kWeightBase = (long) (numHeads + head) * headDim * hidden;
            long vWeightBase = (long) (numHeads + numHeadsKv + head) * headDim * hidden;
            
            long kBiasBase = (long) numHeads * headDim + (long) head * headDim;
            long vBiasBase = (long) (numHeads + numHeadsKv) * headDim + (long) head * headDim;
            
            long kvOutBase = (long) head * headDim;

            for (int d = 0; d < headDim; d++) {
                float sumK = dot(x, wPacked, kWeightBase + (long) d * hidden, hidden);
                if (bqkv != null) sumK += bqkv.get(ValueLayout.JAVA_FLOAT, (kBiasBase + d) * 4L);
                outK.set(ValueLayout.JAVA_FLOAT, (kvOutBase + d) * 4L, sumK);

                float sumV = dot(x, wPacked, vWeightBase + (long) d * hidden, hidden);
                if (bqkv != null) sumV += bqkv.get(ValueLayout.JAVA_FLOAT, (vBiasBase + d) * 4L);
                outV.set(ValueLayout.JAVA_FLOAT, (kvOutBase + d) * 4L, sumV);
            }
        }
    }

    private static float dot(MemorySegment vec, MemorySegment mat, long matOffset, int colSize) {
        int i = 0;
        FloatVector vsum = FloatVector.zero(SPECIES);
        long matByteBase = matOffset * 4L;

        // Vector loop
        for (; i <= colSize - SPECIES.length(); i += SPECIES.length()) {
            var v1 = FloatVector.fromMemorySegment(SPECIES, vec, (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            var v2 = FloatVector.fromMemorySegment(SPECIES, mat, matByteBase + (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            vsum = v1.fma(v2, vsum);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        // Scalar tail
        for (; i < colSize; i++) {
            float val1 = vec.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
            float val2 = mat.get(ValueLayout.JAVA_FLOAT, matByteBase + (long) i * 4L);
            sum += val1 * val2;
        }

        return sum;
    }
}
