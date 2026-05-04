package tech.kayys.gollek.runtime;

import java.lang.foreign.*;

import tech.kayys.gollek.core.tensor.*;
import tech.kayys.gollek.runtime.kv.KVCache;
import tech.kayys.gollek.backend.cpu.FlashAttentionCpu;
import tech.kayys.gollek.core.memory.CpuBuffer;
import tech.kayys.gollek.runtime.kv.KVCache;
import tech.kayys.gollek.runtime.tensor.DefaultTensor;
import tech.kayys.gollek.runtime.tensor.RuntimeTensor;

/**
 * 
 * DECODE (a.k.a. autoregressive):
 * Input: single token [1, D]
 * Output: next token
 * Compute: attention over KV cache
 * Cost: O(T) per token
 * 
 */
public final class DecodeRunner {
        private DecodeRunner() {
        }

        public static Tensor step(
                        Tensor x, // [1, D]
                        Tensor wqkv,
                        KVCache cache,
                        int heads) {
                int dim = (int) x.shape().dim(1);
                int headDim = dim / heads;
                // ---- QKV ----
                Tensor qkv = x.backend().matmul(x, wqkv);
                MemorySegment seg = qkv.buffer().segment();
                MemorySegment Q = seg.asSlice(0, dim * 4);
                MemorySegment K = seg.asSlice(dim * 4, dim * 4);
                MemorySegment V = seg.asSlice(dim * 8, dim * 4);
                // ---- append to cache ----
                cache.append(K, V);
                int seqLen = cache.position();
                // ---- decode cache ----
                Arena arena = Arena.ofAuto();
                MemorySegment kDecoded = arena.allocate((long) seqLen * heads * headDim * 4);
                MemorySegment vDecoded = arena.allocate((long) seqLen * heads * headDim * 4);
                cache.decodeK(kDecoded);
                cache.decodeV(vDecoded);
                // ---- attention ----
                CpuBuffer out = new CpuBuffer(dim * 4);
                FlashAttentionCpu.forwardWithCache(
                                Q,
                                kDecoded,
                                vDecoded, out.segment(),
                                seqLen,
                                heads,
                                headDim);
                return new DefaultTensor(
                                new Shape(1, dim),
                                x.dtype(),
                                x.device(),
                                out,
                                x.backend());
        }
}
