package tech.kayys.gollek.gguf.loader.inference;

public final class KVCache {
    private final float[][] k;
    private final float[][] v;
    private final int maxSeq;
    private final int nKVHeads;
    private final int headDim;
    private final int headStride;
    private int seqLen = 0;
    
    public KVCache(int nLayers, int maxSeq, int nKVHeads, int headDim) {
        this.maxSeq = maxSeq;
        this.nKVHeads = nKVHeads;
        this.headDim = headDim;
        this.headStride = nKVHeads * headDim;
        this.k = new float[nLayers][maxSeq * headStride];
        this.v = new float[nLayers][maxSeq * headStride];
    }
    
    public void store(int layer, float[] kVec, float[] vVec) {
        int base = seqLen * headStride;
        System.arraycopy(kVec, 0, k[layer], base, headStride);
        System.arraycopy(vVec, 0, v[layer], base, headStride);
    }
    
    public void advance() { seqLen++; }
    
    public void readK(int layer, int pos, int kvHead, float[] dst, int dstOff) {
        int src = pos * headStride + kvHead * headDim;
        System.arraycopy(k[layer], src, dst, dstOff, headDim);
    }
    
    public void readV(int layer, int pos, int kvHead, float[] dst, int dstOff) {
        int src = pos * headStride + kvHead * headDim;
        System.arraycopy(v[layer], src, dst, dstOff, headDim);
    }
    
    public float[] kLayer(int layer) { return k[layer]; }
    public float[] vLayer(int layer) { return v[layer]; }
    public int seqLen() { return seqLen; }
    public int headStride() { return headStride; }
    public void reset() { seqLen = 0; java.util.Arrays.fill(k[0], 0f); } // Simplified reset
}
