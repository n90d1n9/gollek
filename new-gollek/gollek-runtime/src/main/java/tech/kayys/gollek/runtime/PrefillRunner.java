package tech.kayys.gollek.runtime;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.runtime.kv.KVCache;
import tech.kayys.gollek.runtime.tensor.RuntimeTensor;

import java.lang.foreign.MemorySegment;

/**
 * 
 * PREFILL (a.k.a. prompt processing)
 * Input: full sequence [T, D]
 * Output: KV cache filled for all tokens
 * Compute: full attention (FlashAttention v2/v3)
 * Cost: O(T²) (but done once)
 * 
 */
public final class PrefillRunner {
    private PrefillRunner() {
    }

    public static void run(
            Tensor x, // [T, D]
            Tensor wqkv,
            KVCache cache,
            int heads) {
        int seq = (int) x.shape().dim(0);
        int dim = (int) x.shape().dim(1);
        // ---- compute QKV for full sequence ----
        Tensor qkv = matmul(x, wqkv); // [T, 3D]
        MemorySegment seg = qkv.buffer().segment();
        long tokenSize = dim * 4;
        for (int t = 0; t < seq; t++) {
            long base = t * dim * 3L;
            MemorySegment K = seg.asSlice((base + dim) * 4, tokenSize);
            MemorySegment V = seg.asSlice((base + dim * 2) * 4, tokenSize);
            cache.append(K, V);
        }
    }

    // minimal matmul hook (replace with your backend)
    private static Tensor matmul(Tensor a, Tensor b) {
        return a.backend().matmul(a, b);
    }
}