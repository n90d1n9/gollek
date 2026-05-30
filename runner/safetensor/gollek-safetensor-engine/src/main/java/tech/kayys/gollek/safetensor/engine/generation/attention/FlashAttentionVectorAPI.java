package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Flash Attention-2 implementation using Java Vector API.
 * Optimized for CPU inference on long contexts.
 */
public class FlashAttentionVectorAPI {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * Compute attention using the Flash Attention algorithm.
     * Output = softmax(Q @ K.T * scale) @ V
     *
     * @param q      Query tensor [B, H_q, S_q, D]
     * @param k      Key tensor [B, H_kv, S_kv, D]
     * @param v      Value tensor [B, H_kv, S_kv, D]
     * @param scale  Scaling factor (1/sqrt(headDim))
     * @param causal Whether to apply causal masking
     * @return Output tensor [B, H_q, S_q, D]
     */
    public static AccelTensor compute(AccelTensor q, AccelTensor k, AccelTensor v, float scale, boolean causal) {
        long batch = q.size(0);
        long numQHeads = q.size(1);
        long seqLenQ = q.size(2);
        long headDim = q.size(3);
        long numKVHeads = k.size(1);
        long seqLenKV = k.size(2);
        int groupSize = (int) (numQHeads / numKVHeads);

        AccelTensor out = AccelTensor.zeros(batch, numQHeads, seqLenQ, headDim);

        for (int b = 0; b < batch; b++) {
            for (int hq = 0; hq < numQHeads; hq++) {
                int hkv = hq / groupSize;
                float[] accScratch = new float[(int) headDim];
                
                for (int i = 0; i < seqLenQ; i++) {
                    computeHeadQuery(b, hq, hkv, i, q, k, v, out, scale, seqLenKV, headDim, causal,
                            accScratch);
                }
            }
        }
        return out;
    }

    private static void computeHeadQuery(int b, int hq, int hkv, int i, 
                                        AccelTensor q, AccelTensor k, AccelTensor v, AccelTensor out,
                                        float scale, long seqLenKV, long headDim, boolean causal, float[] acc) {
        
        MemorySegment qSeg = q.dataSegment();
        MemorySegment kSeg = k.dataSegment();
        MemorySegment vSeg = v.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        long qOff = ((((long)b * q.size(1) + hq) * q.size(2) + i) * headDim) * ValueLayout.JAVA_FLOAT.byteSize();
        
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        Arrays.fill(acc, 0, Math.toIntExact(headDim), 0.0f);

        // Softmax Online Loop
        for (int j = 0; j < seqLenKV; j++) {
            // Causal masking
            if (causal && j > i) break; 

            // 1. Compute Dot Product Q_i @ K_j
            long kOff = ((((long)b * k.size(1) + hkv) * k.size(2) + j) * headDim) * ValueLayout.JAVA_FLOAT.byteSize();
            float score = dotProduct(qSeg, qOff, kSeg, kOff, headDim) * scale;

            // 2. Online Softmax Update
            float m_prev = m;
            m = Math.max(m, score);
            
            float exp_prev = (float) Math.exp(m_prev - m);
            float exp_curr = (float) Math.exp(score - m);
            
            l = l * exp_prev + exp_curr;

            // 3. Update Output Accumulator
            long vOff = ((((long)b * v.size(1) + hkv) * v.size(2) + j) * headDim) * ValueLayout.JAVA_FLOAT.byteSize();
            updateAccumulator(acc, vSeg, vOff, exp_prev, exp_curr, headDim);
        }

        // 4. Final Normalization
        long oOff = ((((long)b * out.size(1) + hq) * out.size(2) + i) * headDim) * ValueLayout.JAVA_FLOAT.byteSize();
        writeNormalizedAccumulator(oSeg, oOff / Float.BYTES, acc, 1.0f / (l + 1e-9f), Math.toIntExact(headDim));
    }

    private static float dotProduct(MemorySegment q, long qOff, MemorySegment k, long kOff, long dim) {
        int n = Math.toIntExact(dim);
        int i = 0;
        FloatVector sum = FloatVector.zero(SPECIES);
        int upperBound = SPECIES.loopBound(n);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector qVec = FloatVector.fromMemorySegment(
                    SPECIES, q, qOff + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            FloatVector kVec = FloatVector.fromMemorySegment(
                    SPECIES, k, kOff + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            sum = sum.add(qVec.mul(kVec));
        }
        float res = sum.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + i) * 
                   k.getAtIndex(ValueLayout.JAVA_FLOAT, (kOff / 4) + i);
        }
        return res;
    }

    private static void updateAccumulator(float[] acc, MemorySegment vSeg, long vOff, float exp_prev, float exp_curr, long dim) {
        int n = Math.toIntExact(dim);
        int i = 0;
        FloatVector prev = FloatVector.broadcast(SPECIES, exp_prev);
        FloatVector curr = FloatVector.broadcast(SPECIES, exp_curr);
        int upperBound = SPECIES.loopBound(n);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector accVec = FloatVector.fromArray(SPECIES, acc, i);
            FloatVector valueVec = FloatVector.fromMemorySegment(
                    SPECIES, vSeg, vOff + (long) i * Float.BYTES, ByteOrder.nativeOrder());
            accVec.mul(prev).add(valueVec.mul(curr)).intoArray(acc, i);
        }
        for (; i < n; i++) {
            acc[i] = acc[i] * exp_prev + vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (vOff / 4) + i) * exp_curr;
        }
    }

    private static void writeNormalizedAccumulator(MemorySegment out, long outIndex, float[] acc, float invL,
            int headDim) {
        int i = 0;
        FloatVector inv = FloatVector.broadcast(SPECIES, invL);
        int upperBound = SPECIES.loopBound(headDim);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, acc, i)
                    .mul(inv)
                    .intoMemorySegment(out, (outIndex + i) * Float.BYTES, ByteOrder.nativeOrder());
        }
        for (; i < headDim; i++) {
            out.setAtIndex(ValueLayout.JAVA_FLOAT, outIndex + i, acc[i] * invL);
        }
    }
}
