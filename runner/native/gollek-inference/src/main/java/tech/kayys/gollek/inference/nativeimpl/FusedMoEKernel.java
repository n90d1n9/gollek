package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;
import tech.kayys.gollek.spi.model.FFNActivationType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fused Mixture-of-Experts (MoE) Kernel.
 * Handles expert routing, softmax gating, and parallel expert dispatch.
 */
public final class FusedMoEKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private FusedMoEKernel() {}

    public static void executeParallel(
        MemorySegment x,
        TransformerLayerWeights w,
        LayerBuffers buf,
        int hidden,
        int ffnDim,
        int numExperts,
        int numExpertsPerTok,
        FFNActivationType activation,
        ExecutorService executor
    ) {
        // 1. Router: moeLogits = x @ gateInp
        for (int e = 0; e < numExperts; e++) {
            float sum = dot(x, w.ffnGateInpWeight, (long) e * hidden, hidden);
            buf.moeLogits.set(ValueLayout.JAVA_FLOAT, (long) e * 4L, sum);
        }

        // 2. Softmax(moeLogits) -> moeWeights
        softmax(buf.moeLogits, buf.moeWeights, numExperts);

        // 3. Find Top-K experts
        int[] topKIndices = findTopK(buf.moeWeights, numExperts, numExpertsPerTok);
        
        // 4. Renormalize top-K weights
        float weightSum = 0;
        for (int i = 0; i < numExpertsPerTok; i++) {
            weightSum += buf.moeWeights.get(ValueLayout.JAVA_FLOAT, (long) topKIndices[i] * 4L);
        }
        
        // 5. Run experts and accumulate
        buf.ffnExpertOut.fill((byte)0);
        for (int i = 0; i < numExpertsPerTok; i++) {
            int expertIdx = topKIndices[i];
            float gateWeight = buf.moeWeights.get(ValueLayout.JAVA_FLOAT, (long) expertIdx * 4L);
            if (weightSum > 0) gateWeight /= weightSum;
            
            FusedFFNKernel.computeParallel(
                x, 
                w.wGExperts[expertIdx], null,
                w.wUExperts[expertIdx], null,
                w.wDExperts[expertIdx], null,
                buf.ffn,
                buf.ffnExpertOut, buf.ffnExpertOut, // Accumulate
                hidden, ffnDim, activation, gateWeight, executor
            );
        }
    }

    private static float dot(MemorySegment vec, MemorySegment mat, long matOffset, int colSize) {
        int i = 0;
        FloatVector vsum = FloatVector.zero(SPECIES);
        long matByteBase = matOffset * 4L;

        for (; i <= colSize - SPECIES.length(); i += SPECIES.length()) {
            var v1 = FloatVector.fromMemorySegment(SPECIES, vec, (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            var v2 = FloatVector.fromMemorySegment(SPECIES, mat, matByteBase + (long) i * 4L, java.nio.ByteOrder.nativeOrder());
            vsum = v1.fma(v2, vsum);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        for (; i < colSize; i++) {
            float val1 = vec.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
            float val2 = mat.get(ValueLayout.JAVA_FLOAT, matByteBase + (long) i * 4L);
            sum += val1 * val2;
        }

        return sum;
    }

    private static void softmax(MemorySegment logits, MemorySegment weights, int n) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            float val = logits.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
            if (val > max) max = val;
        }

        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float val = (float) Math.exp(logits.get(ValueLayout.JAVA_FLOAT, (long) i * 4L) - max);
            weights.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, val);
            sum += val;
        }

        for (int i = 0; i < n; i++) {
            float val = weights.get(ValueLayout.JAVA_FLOAT, (long) i * 4L) / sum;
            weights.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, val);
        }
    }

    private static int[] findTopK(MemorySegment weights, int n, int k) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        // Simple selection sort for top-K (usually K is very small, like 2 or 8)
        for (int i = 0; i < k; i++) {
            int maxIdx = i;
            float maxVal = weights.get(ValueLayout.JAVA_FLOAT, (long) indices[i] * 4L);
            for (int j = i + 1; j < n; j++) {
                float val = weights.get(ValueLayout.JAVA_FLOAT, (long) indices[j] * 4L);
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            int temp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = temp;
        }

        int[] result = new int[k];
        System.arraycopy(indices, 0, result, 0, k);
        return result;
    }
}
