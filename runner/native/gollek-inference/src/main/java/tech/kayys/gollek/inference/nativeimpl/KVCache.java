package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Append-only, zero-copy friendly Key-Value Cache.
 * Memory layout is flattened as [layer][head][head_dim][max_seq_len]
 * to ensure that for a fixed dimension, values across the sequence are contiguous.
 * This allows vectorizing across the 'pos' dimension.
 */
public final class KVCache {

    private final MemorySegment kCache;
    private final MemorySegment vCache;

    private final int numLayers;
    private final int numHeadsKv;
    private final int headDim;
    private final int maxSeqLen;
    private final int qHeadsPerKvHead; // for GQA mapping

    public KVCache(
            int numLayers,
            int numHeads,
            int numHeadsKv,
            int headDim,
            int maxSeqLen,
            MemorySegment kCache,
            MemorySegment vCache
    ) {
        this.numLayers = numLayers;
        this.numHeadsKv = numHeadsKv;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.qHeadsPerKvHead = numHeads / numHeadsKv;
        this.kCache = kCache;
        this.vCache = vCache;
    }

    public long offset(int layer, int qHead, int dim, int pos) {
        int kvHead = qHead / qHeadsPerKvHead;
        return (((long)layer * numHeadsKv + kvHead) * headDim + dim) * maxSeqLen + pos;
    }

    public float getK(int l, int h, int p, int d) {
        long idx = offset(l, h, d, p) * Float.BYTES;
        return kCache.get(ValueLayout.JAVA_FLOAT, idx);
    }

    public void setK(int l, int h, int p, int d, float v) {
        long idx = offset(l, h, d, p) * Float.BYTES;
        kCache.set(ValueLayout.JAVA_FLOAT, idx, v);
    }

    public float getV(int l, int h, int p, int d) {
        long idx = offset(l, h, d, p) * Float.BYTES;
        return vCache.get(ValueLayout.JAVA_FLOAT, idx);
    }

    public void setV(int l, int h, int p, int d, float v) {
        long idx = offset(l, h, d, p) * Float.BYTES;
        vCache.set(ValueLayout.JAVA_FLOAT, idx, v);
    }

    /**
     * Note: In this layout, a 'Row' across dims for a fixed pos is STRIDED.
     * To get a contiguous row for a fixed dim across pos, use getKSequenceVector.
     */
    public MemorySegment getKRow(int layer, int head, int pos) {
        // Warning: This is now strided! Calling asSlice won't work for a full row.
        // We'll keep it for API compatibility if needed, but it only returns [layer, head, 0, pos].
        return kCache.asSlice(offset(layer, head, 0, pos) * Float.BYTES, Float.BYTES);
    }

    public MemorySegment getVRow(int layer, int head, int pos) {
        return vCache.asSlice(offset(layer, head, 0, pos) * Float.BYTES, Float.BYTES);
    }

    public MemorySegment getKCache() { return kCache; }
    public MemorySegment getVCache() { return vCache; }
}
