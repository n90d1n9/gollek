package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.gguf.loader.TransformerLayerWeights;

/**
 * Executes a single transformer layer end-to-end with zero intermediary allocations.
 */
public final class FusedTransformerLayer {

    private FusedTransformerLayer() {}

    public static void execute(
        int layerIdx,
        int pos,
        MemorySegment x,
        MemorySegment xNext,
        TransformerLayerWeights w,
        LayerBuffers buf,
        KVCache kvCache,
        RoPECache ropeCache,
        int hidden,
        int numHeads,
        int numHeadsKv,
        int headDim,
        boolean isNeox,
        float eps,
        float attnSoftCap,
        boolean addOneToWeight,
        tech.kayys.gollek.spi.model.FFNActivationType activation,
        ExecutorService executor
    ) {
        // 1. RMSNorm (Attention) -> buf.norm
        RMSNormKernel.execute(x, buf.norm, w.rmsWeight, hidden, eps, addOneToWeight);
        
        // 2. QKV Projection -> buf.q, buf.k, buf.v (handles QKV biases)
        PrepackedQKVKernel.computeParallel(buf.norm, w.wqkvPacked, w.bqkv, buf.q, buf.k, buf.v, hidden, numHeads, numHeadsKv, headDim, executor);

        // 3. Apply RoPE
        for (int h = 0; h < numHeads; h++) {
            MemorySegment qHead = buf.q.asSlice((long) h * headDim * Float.BYTES, (long) headDim * Float.BYTES);
            RoPEKernel.apply(qHead, pos, ropeCache, headDim, isNeox);
        }
        for (int h = 0; h < numHeadsKv; h++) {
            MemorySegment kHead = buf.k.asSlice((long) h * headDim * Float.BYTES, (long) headDim * Float.BYTES);
            RoPEKernel.apply(kHead, pos, ropeCache, headDim, isNeox);
        }

        // 4. Update KV Cache
        int qHeadsPerKvHead = numHeads / numHeadsKv;
        for (int h = 0; h < numHeadsKv; h++) {
            for (int d = 0; d < headDim; d++) {
                float kval = buf.k.get(ValueLayout.JAVA_FLOAT, (long)((long) h * headDim + d) * Float.BYTES);
                float vval = buf.v.get(ValueLayout.JAVA_FLOAT, (long)((long) h * headDim + d) * Float.BYTES);
                // We use the first Q head of the group as the storage key
                kvCache.setK(layerIdx, h * qHeadsPerKvHead, pos, d, kval);
                kvCache.setV(layerIdx, h * qHeadsPerKvHead, pos, d, vval);
            }
        }

        // 5. FlashAttention -> buf.attnOut
        FlashAttentionKernel.computeParallel(buf.attnOut, buf.q, kvCache, null, layerIdx, pos + 1, numHeads, headDim, attnSoftCap, executor);

        // 6. Output Projection (handles Output bias)
        // inDim = numHeads * headDim (attention output size), outDim = hidden
        int attnOutDim = numHeads * headDim;
        if (w.postAttnNormWeight != null) {
            LinearResidualKernel.computeParallel(buf.attnOut, w.wo, w.bo, null, buf.norm, attnOutDim, hidden, executor);
            RMSNormKernel.execute(buf.norm, buf.norm, w.postAttnNormWeight, hidden, eps, addOneToWeight);
            addInPlace(x, buf.norm, hidden);
        } else {
            LinearResidualKernel.computeParallel(buf.attnOut, w.wo, w.bo, x, x, attnOutDim, hidden, executor);
        }

        // 7. RMSNorm (FFN) -> buf.norm
        RMSNormKernel.execute(x, buf.norm, w.ffnNormWeight, hidden, eps, addOneToWeight);

        // 8. FFN
        int ffnDim = (int) (buf.ffn.byteSize() / Float.BYTES);
        if (w.postFfnNormWeight != null) {
            FusedFFNKernel.computeParallel(buf.norm, w, buf.ffn, null, buf.norm, hidden, ffnDim, activation, executor);
            RMSNormKernel.execute(buf.norm, buf.norm, w.postFfnNormWeight, hidden, eps, addOneToWeight);
            addInPlace(x, buf.norm, hidden);
            MemorySegment.copy(x, 0, xNext, 0, x.byteSize()); // Ensure xNext has the result
        } else {
            FusedFFNKernel.computeParallel(buf.norm, w, buf.ffn, x, xNext, hidden, ffnDim, activation, executor);
        }
    }

    private static void addInPlace(MemorySegment dest, MemorySegment src, int size) {
        int i = 0;
        jdk.incubator.vector.VectorSpecies<Float> SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
        for (; i <= size - SPECIES.length(); i += SPECIES.length()) {
            var vd = jdk.incubator.vector.FloatVector.fromMemorySegment(SPECIES, dest, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            var vs = jdk.incubator.vector.FloatVector.fromMemorySegment(SPECIES, src, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            vd.add(vs).intoMemorySegment(dest, (long) i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < size; i++) {
            float d = dest.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            float s = src.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            dest.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, d + s);
        }
    }
}
