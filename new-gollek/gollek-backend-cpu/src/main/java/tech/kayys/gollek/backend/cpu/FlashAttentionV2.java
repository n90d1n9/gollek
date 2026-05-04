package tech.kayys.gollek.backend.cpu;

import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.concurrent.*;

public final class FlashAttentionV2 {
    private static final jdk.incubator.vector.VectorSpecies<Float> SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
    private static final int BLOCK_Q = 64;
    private static final int BLOCK_K = 64;

    public static void forward(
            MemorySegment Q,
            MemorySegment K,
            MemorySegment V,
            MemorySegment O,
            int seqLen,
            int dim,
            int numThreads) throws InterruptedException {
        ExecutorService pool = Executors.newFixedThreadPool(numThreads);
        int numBlocks = (seqLen + BLOCK_Q - 1) / BLOCK_Q;
        CountDownLatch latch = new CountDownLatch(numBlocks);
        for (int qb = 0; qb < numBlocks; qb++) {
            final int qStart = qb * BLOCK_Q;
            final int qEnd = Math.min(qStart + BLOCK_Q, seqLen);
            pool.submit(() -> {
                computeQBlock(Q, K, V, O, qStart, qEnd, seqLen, dim);
                latch.countDown();
            });
        }
        latch.await();
        pool.shutdown();
    }

    private static void computeQBlock(
            MemorySegment Q,
            MemorySegment K,
            MemorySegment V,
            MemorySegment O,
            int qStart,
            int qEnd,
            int seqLen,
            int dim) {
        float scale = (float) (1.0 / Math.sqrt(dim));
        for (int i = qStart; i < qEnd; i++) {
            float m_i = Float.NEGATIVE_INFINITY;
            float l_i = 0f;
            float[] acc = new float[dim];
            // iterate K in blocks
            for (int kb = 0; kb < seqLen; kb += BLOCK_K) {
                int kEnd = Math.min(kb + BLOCK_K, seqLen);
                for (int j = kb; j < kEnd; j++) {
                    float score = dot(Q, K, i, j, dim) * scale;
                    float m_new = Math.max(m_i, score);
                    float exp_old = (float) Math.exp(m_i - m_new);
                    float exp_new = (float) Math.exp(score - m_new);
                    float l_new = l_i * exp_old + exp_new;
                    float alpha = (l_i == 0f) ? 0f : (l_i * exp_old / l_new);
                    float beta = exp_new / l_new;
                    // vectorized update (v3-style improvement)
                    updateAccumulator(acc, V, j, dim, alpha, beta);
                    m_i = m_new;
                    l_i = l_new;
                }
            }
            // write output
            for (int d = 0; d < dim; d++) {
                O.set(ValueLayout.JAVA_FLOAT,
                        ((long) i * dim + d) * 4, acc[d]);
            }
        }
    }

    // VECTOR ACCUMULATOR UPDATE (V3 STYLE)
    private static void updateAccumulator(
            float[] acc,
            MemorySegment V,
            int j,
            int dim,
            float alpha,
            float beta) {
        int stride = SPECIES.length();
        int d = 0;
        long base = (long) j * dim;
        for (; d <= dim - stride; d += stride) {
            var vacc = jdk.incubator.vector.FloatVector.fromArray(SPECIES, acc, d);
            var vv = jdk.incubator.vector.FloatVector.fromMemorySegment(
                    SPECIES, V, (base + d) * 4, ByteOrder.nativeOrder());
            vacc = vacc.mul(alpha).add(vv.mul(beta));
            vacc.intoArray(acc, d);
        }
        for (; d < dim; d++) {
            float v = V.get(ValueLayout.JAVA_FLOAT, (base + d) * 4);
            acc[d] = acc[d] * alpha + v * beta;
        }
    }

/**
 * Layout:
[B, H, T, D]
Strategy
parallel over (batch * heads * Q-blocks)
no cross-thread sync
perfect scaling
:
 * @param Q
 * @param K
 * @param V
 * @param O
 * @param batch
 * @param heads
 * @param seqLen
 * @param dim
 */
public static void forwardMultiHead(
MemorySegment Q,
MemorySegment K,
MemorySegment V,
MemorySegment O,
int batch,
int heads,
int seqLen,
int dim
)

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        int seq = (int) Q.shape().dim(0);
        int dim = (int) Q.shape().dim(1);
        CpuBuffer out = new CpuBuffer((long) seq * dim * 4);
        FlashAttentionV2.forward(
                Q.buffer().segment(),
                K.buffer().segment(),
                V.buffer().segment(),
                out.segment(),
                seq,
                dim,
                Runtime.getRuntime().availableProcessors());
        return new DefaultTensor(
                new Shape(seq, dim),
                Q.dtype(),
                Q.device(),
                out,
                this);
    }

private static void updateAccumulatorFullyVectorized(
        float[] acc,
        MemorySegment V,
        int j,
        int dim,
        float alpha,
        float beta) {
    
    int stride = SPECIES.length();
    int d = 0;
    long base = (long) j * dim;
    
    // Vectorized path for all elements
    for (; d <= dim - stride; d += stride) {
        FloatVector vacc = FloatVector.fromArray(SPECIES, acc, d);
        FloatVector vv = FloatVector.fromMemorySegment(
            SPECIES, V, (base + d) * 4, ByteOrder.nativeOrder());
        
        // Fused multiply-add for better performance
        vacc = vacc.fma(vv, FloatVector.broadcast(SPECIES, beta));
        vacc = vacc.fma(FloatVector.broadcast(SPECIES, alpha), 
                        FloatVector.broadcast(SPECIES, 1.0f));
        
        vacc.intoArray(acc, d);
    }
    
    // Scalar remainder - but now only for < stride elements
    for (; d < dim; d++) {
        float v = V.get(ValueLayout.JAVA_FLOAT, (base + d) * 4);
        acc[d] = acc[d] * alpha + v * beta;
    }
}