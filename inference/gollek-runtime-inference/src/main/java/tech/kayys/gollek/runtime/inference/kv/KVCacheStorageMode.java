package tech.kayys.gollek.runtime.inference.kv;

/**
 * Storage mode for KV cache pages.
 * <p>
 * Determines how K/V tensors are stored in paged memory blocks.
 * Different modes trade off memory usage vs compute overhead.
 *
 * <h2>Storage Modes</h2>
 *
 * <table>
 *   <tr><th>Mode</th><th>Bits/Token</th><th>Memory (70B, 128ctx)</th><th>Accuracy</th><th>Latency Overhead</th></tr>
 *   <tr><td>FULL_PRECISION</td><td>16-bit (FP16)</td><td>280 GB</td><td>1.000</td><td>0%</td></tr>
 *   <tr><td>TURBOQUANT_4BIT</td><td>4-bit</td><td>70 GB</td><td>0.999</td><td>~10%</td></tr>
 *   <tr><td>TURBOQUANT_3BIT</td><td>3-bit</td><td>47 GB</td><td>0.997</td><td>~15%</td></tr>
 *   <tr><td>TURBOQUANT_2BIT</td><td>2.5-bit (outlier split)</td><td>35 GB</td><td>0.995</td><td>~20%</td></tr>
 * </table>
 *
 * <h2>Recommendations</h2>
 * <ul>
 *   <li><b>FULL_PRECISION:</b> Maximum accuracy, small context, research</li>
 *   <li><b>TURBOQUANT_4BIT:</b> Production default, best quality/size tradeoff</li>
 *   <li><b>TURBOQUANT_3BIT:</b> High-throughput serving, 6× compression</li>
 *   <li><b>TURBOQUANT_2BIT:</b> Maximum density, multi-tenant, long context</li>
 * </ul>
 *
 * @see PagedKVCache
 * @see TurboQuantKVCacheAdapter
 * @since 0.2.0
 */
public enum KVCacheStorageMode {

    /**
     * Full precision FP16/FP32 storage.
     * <p>
     * No compression. Highest accuracy and lowest latency.
     * Best for research, small context, or latency-sensitive workloads.
     */
    FULL_PRECISION,

    /**
     * TurboQuant 4-bit compression.
     * <p>
     * Uses TurboQuant_prod with 4-bit MSE + QJL residual.
     * Compression: ~4× vs FP16.
     * Accuracy: >0.999 correlation with full precision.
     */
    TURBOQUANT_4BIT,

    /**
     * TurboQuant 3-bit compression (recommended for production serving).
     * <p>
     * Uses TurboQuant_prod with 3-bit MSE + QJL residual.
     * Compression: ~6× vs FP16.
     * Accuracy: >0.997 correlation with full precision (NIAH score).
     */
    TURBOQUANT_3BIT,

    /**
     * TurboQuant 2.5-bit compression with outlier channel splitting.
     * <p>
     * Uses TurboQuant_prod with outlier splitting:
     * 32 channels × 3-bit + 96 channels × 2-bit = 2.5-bit effective.
     * Compression: ~8× vs FP16.
     * Accuracy: >0.995 correlation with full precision.
     */
    TURBOQUANT_2BIT;

    /**
     * Returns whether this mode uses TurboQuant compression.
     */
    public boolean isCompressed() {
        return this != FULL_PRECISION;
    }

    /**
     * Returns the effective bits per token for this mode.
     */
    public int bitsPerToken() {
        return switch (this) {
            case FULL_PRECISION -> 16;
            case TURBOQUANT_4BIT -> 4;
            case TURBOQUANT_3BIT -> 3;
            case TURBOQUANT_2BIT -> 3; // 2.5 rounded up
        };
    }

    /**
     * Returns the compression ratio vs FP16.
     *
     * @return ratio (e.g., 6.0 means 6× smaller than FP16)
     */
    public double compressionRatio() {
        return switch (this) {
            case FULL_PRECISION -> 1.0;
            case TURBOQUANT_4BIT -> 4.0;
            case TURBOQUANT_3BIT -> 6.0;
            case TURBOQUANT_2BIT -> 8.0;
        };
    }

    /**
     * Returns the expected accuracy correlation vs full precision.
     * Based on TurboQuant paper benchmarks.
     */
    public double expectedAccuracy() {
        return switch (this) {
            case FULL_PRECISION -> 1.0;
            case TURBOQUANT_4BIT -> 0.999;
            case TURBOQUANT_3BIT -> 0.997;
            case TURBOQUANT_2BIT -> 0.995;
        };
    }

    /**
     * Returns the expected latency overhead.
     */
    public double expectedLatencyOverhead() {
        return switch (this) {
            case FULL_PRECISION -> 0.0;
            case TURBOQUANT_4BIT -> 0.10;
            case TURBOQUANT_3BIT -> 0.15;
            case TURBOQUANT_2BIT -> 0.20;
        };
    }

    /**
     * Returns whether this mode is suitable for production serving.
     * (Accuracy > 0.995)
     */
    public boolean isProductionReady() {
        return expectedAccuracy() >= 0.995;
    }

    /**
     * Gets the recommended mode for a use case.
     *
     * @param useCase one of: "research", "production", "high-throughput", "maximum-density"
     * @return recommended storage mode
     */
    public static KVCacheStorageMode recommend(String useCase) {
        return switch (useCase.toLowerCase()) {
            case "research", "maximum-accuracy" -> FULL_PRECISION;
            case "production", "balanced" -> TURBOQUANT_3BIT;
            case "high-throughput" -> TURBOQUANT_3BIT;
            case "maximum-density", "multi-tenant" -> TURBOQUANT_2BIT;
            default -> TURBOQUANT_3BIT; // Safe default
        };
    }
}
