package tech.kayys.gollek.runtime.inference.kv;

import tech.kayys.gollek.runtime.tensor.Tensor;
import tech.kayys.gollek.runtime.tensor.ExecutionContext;

/**
 * PagedAttention kernel interface.
 * <p>
 * This defines the contract for attention kernels that operate on paged
 * KV caches. Instead of requiring contiguous K/V tensors, these kernels
 * read directly from the scattered block memory via block tables.
 * <p>
 * <h2>How PagedAttention Works</h2>
 * <pre>
 * Standard Attention:
 *   Q @ K^T / sqrt(d) → softmax → @ V
 *   Where K = [K_0, K_1, K_2, ..., K_n] (contiguous)
 *
 * Paged Attention:
 *   For each token position i:
 *     block_idx = blockTable[i / pageSize]
 *     offset = i % pageSize
 *     K_i = read from block[block_idx] at offset
 *   Then compute attention normally
 * </pre>
 *
 * <h2>Benefits</h2>
 * <ul>
 *   <li><b>No Memory Fragmentation:</b> Blocks are fixed-size, eliminating fragmentation</li>
 *   <li><b>Efficient Sharing:</b> Multiple sequences can share prefix blocks</li>
 *   <li><b>Dynamic Batching:</b> Different sequences can have different lengths in same batch</li>
 *   <li><b>Memory Efficiency:</b> Only allocate blocks as needed, not max length</li>
 * </ul>
 *
 * <h2>Implementation</h2>
 * <p>
 * Concrete implementations should use FFM to read directly from the native
 * memory segments in {@link PagedKVCache} blocks. For GPU backends, the
 * block table is uploaded to GPU memory and the attention kernel uses
 * indirect indexing.
 *
 * @see PagedKVCache
 * @see ContinuousBatchScheduler
 * @since 0.2.0
 */
public interface PagedAttentionKernel {

    /**
     * Computes multi-head attention with paged KV cache.
     * <p>
     * This is the core operation for transformer decode. Given a query tensor
     * and a block table, it computes attention over all cached tokens without
     * requiring contiguous memory.
     *
     * @param query current token query tensor: shape [batchSize, numHeads, headDim]
     * @param cache sequence KV cache containing block table
     * @param layer attention layer index
     * @param ctx execution context for memory management
     * @return attention output tensor: shape [batchSize, numHeads * headDim]
     */
    Tensor forward(
        Tensor query,
        PagedKVCache.SequenceKVCache cache,
        int layer,
        ExecutionContext ctx
    );

    /**
     * Computes attention with explicit block table and causal mask.
     * <p>
     * This variant allows custom mask patterns (e.g., sliding window,
     * prefix LM, etc.).
     *
     * @param query query tensor: shape [batchSize, numHeads, headDim]
     * @param blockTable physical block indices for each layer
     * @param numTokens number of tokens in the sequence (for causal mask)
     * @param layer attention layer index
     * @param ctx execution context
     * @return attention output tensor
     */
    Tensor forward(
        Tensor query,
        int[] blockTable,
        int numTokens,
        int layer,
        ExecutionContext ctx
    );

    /**
     * Computes flash attention with paged KV cache.
     * <p>
     * Flash attention computes attention in tiles to avoid materializing
     * the full N×N attention matrix. This version works with paged blocks.
     *
     * @param query query tensor
     * @param cache sequence KV cache
     * @param layer attention layer index
     * @param ctx execution context
     * @return attention output tensor
     */
    Tensor flashAttention(
        Tensor query,
        PagedKVCache.SequenceKVCache cache,
        int layer,
        ExecutionContext ctx
    );

    /**
     * Computes grouped-query attention (GQA).
     * <p>
     * GQA uses fewer KV heads than query heads, with query heads sharing
     * KV heads. This is used by LLaMA 2/3 and many other models.
     *
     * @param query query tensor: shape [batchSize, numQueryHeads, headDim]
     * @param cache sequence KV cache (with fewer KV heads)
     * @param layer attention layer index
     * @param numQueryHeads number of query heads
     * @param numKVHeads number of KV heads (must divide numQueryHeads)
     * @param ctx execution context
     * @return attention output tensor
     */
    Tensor groupedQueryAttention(
        Tensor query,
        PagedKVCache.SequenceKVCache cache,
        int layer,
        int numQueryHeads,
        int numKVHeads,
        ExecutionContext ctx
    );

    /**
     * Gets the kernel name for logging/metrics.
     */
    String kernelName();
}
