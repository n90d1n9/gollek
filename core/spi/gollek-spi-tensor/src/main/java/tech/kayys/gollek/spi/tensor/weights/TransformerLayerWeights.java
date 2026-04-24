package tech.kayys.gollek.spi.tensor.weights;

import java.lang.foreign.MemorySegment;

/**
 * Structural wrapper pointing to the zero-copy, read-only 
 * FFM MemorySegment instances containing prepacked layer weights.
 */
public final class TransformerLayerWeights {

    public final MemorySegment rmsWeight;        // [hidden]

    public final MemorySegment wqkvPacked;       // [3][heads][headDim][hidden]
    public final MemorySegment bqkv;             // [heads * headDim + 2 * headKv * headDim]

    public final TensorData wo;               // [hidden, hidden]
    public final MemorySegment bo;               // [hidden] (Bias for output projection)

    public final MemorySegment ffnNormWeight;    // [hidden]
    public final MemorySegment postAttnNormWeight; // [hidden] Optional for Gemma 2
    public final MemorySegment postFfnNormWeight;  // [hidden] Optional for Gemma 2
    public final MemorySegment attnQNormWeight;    // [headDim] Gemma 2
    public final MemorySegment attnKNormWeight;    // [headDim] Gemma 2

    public final TensorData wG;               // [ffnDim, hidden]
    public final MemorySegment bg;               // [ffnDim] (Bias for gate projection)
    public final TensorData wU;               // [ffnDim, hidden]
    public final MemorySegment bu;               // [ffnDim] (Bias for up projection)
    public final TensorData wD;               // [hidden, ffnDim]
    public final MemorySegment bd;               // [hidden] (Bias for down projection)

    // MoE specific weights
    public final MemorySegment ffnGateInpWeight; // [num_experts, hidden] -> router
    public final TensorData[] wGExperts;         // [num_experts] of [ffnDim, hidden]
    public final TensorData[] wUExperts;         // [num_experts] of [ffnDim, hidden]
    public final TensorData[] wDExperts;         // [num_experts] of [hidden, ffnDim]

    public TransformerLayerWeights(
            MemorySegment rmsWeight,
            MemorySegment wqkvPacked,
            MemorySegment bqkv,
            TensorData wo,
            MemorySegment bo,
            MemorySegment ffnNormWeight,
            MemorySegment postAttnNormWeight,
            MemorySegment postFfnNormWeight,
            MemorySegment attnQNormWeight,
            MemorySegment attnKNormWeight,
            TensorData wG,
            MemorySegment bg,
            TensorData wU,
            MemorySegment bu,
            TensorData wD,
            MemorySegment bd,
            MemorySegment ffnGateInpWeight,
            TensorData[] wGExperts,
            TensorData[] wUExperts,
            TensorData[] wDExperts) {
        this.rmsWeight = rmsWeight;
        this.wqkvPacked = wqkvPacked;
        this.bqkv = bqkv;
        this.wo = wo;
        this.bo = bo;
        this.ffnNormWeight = ffnNormWeight;
        this.postAttnNormWeight = postAttnNormWeight;
        this.postFfnNormWeight = postFfnNormWeight;
        this.attnQNormWeight = attnQNormWeight;
        this.attnKNormWeight = attnKNormWeight;
        this.wG = wG;
        this.bg = bg;
        this.wU = wU;
        this.bu = bu;
        this.wD = wD;
        this.bd = bd;
        this.ffnGateInpWeight = ffnGateInpWeight;
        this.wGExperts = wGExperts;
        this.wUExperts = wUExperts;
        this.wDExperts = wDExperts;
    }

    public TransformerLayerWeights dequantize(java.lang.foreign.Arena arena) {
        TensorData[] dqWG = null, dqWU = null, dqWD = null;
        if (wGExperts != null) {
            dqWG = new TensorData[wGExperts.length];
            dqWU = new TensorData[wUExperts.length];
            dqWD = new TensorData[wDExperts.length];
            for (int i = 0; i < wGExperts.length; i++) {
                dqWG[i] = wGExperts[i] != null ? wGExperts[i].dequantize(arena) : null;
                dqWU[i] = wUExperts[i] != null ? wUExperts[i].dequantize(arena) : null;
                dqWD[i] = wDExperts[i] != null ? wDExperts[i].dequantize(arena) : null;
            }
        }

        return new TransformerLayerWeights(
            rmsWeight,
            wqkvPacked,
            bqkv,
            wo != null ? wo.dequantize(arena) : null,
            bo,
            ffnNormWeight,
            postAttnNormWeight,
            postFfnNormWeight,
            attnQNormWeight,
            attnKNormWeight,
            wG != null ? wG.dequantize(arena) : null,
            bg,
            wU != null ? wU.dequantize(arena) : null,
            bu,
            wD != null ? wD.dequantize(arena) : null,
            bd,
            ffnGateInpWeight,
            dqWG,
            dqWU,
            dqWD
        );
    }
}
