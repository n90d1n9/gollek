package tech.kayys.gollek.ml.runner;

/**
 * Performance statistics for a model runner.
 *
 * @since 0.3.0
 */
public record PerformanceStats(
    /** Total inference calls */
    long totalCalls,

    /** Total tokens processed (input + output) */
    long totalTokens,

    /** Average latency in milliseconds */
    double avgLatencyMs,

    /** P50 (median) latency in milliseconds */
    double p50LatencyMs,

    /** P95 latency in milliseconds */
    double p95LatencyMs,

    /** P99 latency in milliseconds */
    double p99LatencyMs,

    /** Minimum latency in milliseconds */
    double minLatencyMs,

    /** Maximum latency in milliseconds */
    double maxLatencyMs,

    /** Successful inferences */
    long successes,

    /** Failed inferences */
    long failures
) {
    /**
     * Gets throughput in tokens per second.
     */
    public double tokensPerSecond() {
        double totalLatencySec = avgLatencyMs * totalCalls / 1000.0;
        return totalLatencySec > 0 ? totalTokens / totalLatencySec : 0.0;
    }

    /**
     * Gets success rate as a percentage.
     */
    public double successRate() {
        long total = successes + failures;
        return total > 0 ? (double) successes / total * 100.0 : 100.0;
    }

    /**
     * Gets total inference time in seconds.
     */
    public double totalInferenceTimeSec() {
        return avgLatencyMs * totalCalls / 1000.0;
    }

    /**
     * Empty stats for testing/stubbing.
     */
    public static PerformanceStats empty() {
        return new PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    @Override
    public String toString() {
        return String.format(
            "PerformanceStats{calls=%d, tokens=%d, avg=%.1fms, p95=%.1fms, p99=%.1fms, success=%.1f%%}",
            totalCalls, totalTokens, avgLatencyMs, p95LatencyMs, p99LatencyMs, successRate()
        );
    }
}
