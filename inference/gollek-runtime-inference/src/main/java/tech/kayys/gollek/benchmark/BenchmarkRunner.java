package tech.kayys.gollek.benchmark;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Benchmark framework for comparing Gollek vs vLLM, TGI, ONNX Runtime.
 * <p>
 * Provides standardized benchmarking methodology:
 * <ul>
 *   <li><b>Throughput:</b> Tokens/sec, Requests/sec</li>
 *   <li><b>Latency:</b> P50, P95, P99, TTFT</li>
 *   <li><b>Memory:</b> VRAM usage, concurrent capacity</li>
 *   <li><b>Scalability:</b> Scaling with batch size, GPU count</li>
 * </ul>
 *
 * <h2>Benchmark Types</h2>
 * <pre>
 * 1. Single Request Latency: Measure TTFT, token latency for 1 request
 * 2. Batch Throughput: Max tokens/sec at various batch sizes
 * 3. Concurrent Capacity: Max concurrent requests before OOM/timeout
 * 4. Long Context: Performance with 4K, 8K, 16K, 32K context
 * 5. Multi-Tenant: Performance with rate limiting, canary, observability
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * BenchmarkRunner runner = BenchmarkRunner.builder()
 *     .backend(BenchmarkBackend.GOLLEK)
 *     .model("llama-3-70b")
 *     .warmupIterations(5)
 *     .benchmarkIterations(20)
 *     .batchSizes(List.of(1, 4, 16, 64, 128))
 *     .contextLengths(List.of(128, 512, 2048, 4096))
 *     .build();
 *
 * BenchmarkResults results = runner.runAll();
 *
 * // Compare with other backends
 * BenchmarkResults vllm = runner.withBackend(BenchmarkBackend.VLLM).runAll();
 * BenchmarkResults tgi = runner.withBackend(BenchmarkBackend.TGI).runAll();
 *
 * // Generate comparison report
 * BenchmarkReport report = BenchmarkReport.compare(List.of(results, vllm, tgi));
 * report.printTable();
 * }</pre>
 *
 * @since 0.5.0
 */
public final class BenchmarkRunner {

    private static final Logger LOG = Logger.getLogger(BenchmarkRunner.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Backend under test */
    private final BenchmarkBackend backend;

    /** Model identifier */
    private final String model;

    /** Number of warmup iterations */
    private final int warmupIterations;

    /** Number of benchmark iterations */
    private final int benchmarkIterations;

    /** Batch sizes to test */
    private final List<Integer> batchSizes;

    /** Context lengths to test */
    private final List<Integer> contextLengths;

    /** Maximum tokens to generate per request */
    private final int maxTokens;

    /** Request timeout */
    private final Duration requestTimeout;

    // ── State ─────────────────────────────────────────────────────────

    /** Whether runner is initialized */
    private volatile boolean initialized;

    /** Backend client (would connect to actual service) */
    private Object backendClient;

    private BenchmarkRunner(Config config) {
        this.backend = config.backend;
        this.model = config.model;
        this.warmupIterations = config.warmupIterations;
        this.benchmarkIterations = config.benchmarkIterations;
        this.batchSizes = List.copyOf(config.batchSizes);
        this.contextLengths = List.copyOf(config.contextLengths);
        this.maxTokens = config.maxTokens;
        this.requestTimeout = config.requestTimeout;
    }

    /**
     * Creates a builder for configuring this runner.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Benchmark Execution ───────────────────────────────────────────

    /**
     * Runs all benchmarks and returns results.
     */
    public BenchmarkResults runAll() {
        LOG.infof("Starting benchmark: backend=%s, model=%s, warmup=%d, iterations=%d",
            backend, model, warmupIterations, benchmarkIterations);

        BenchmarkResults.Builder resultsBuilder = BenchmarkResults.builder(backend, model);

        // Run each benchmark type
        resultsBuilder.throughput(runThroughputBenchmark());
        resultsBuilder.latency(runLatencyBenchmark());
        resultsBuilder.concurrency(runConcurrencyBenchmark());
        resultsBuilder.memory(runMemoryBenchmark());
        resultsBuilder.contextScaling(runContextBenchmark());

        BenchmarkResults results = resultsBuilder.build();

        LOG.infof("Benchmark complete: backend=%s, throughput=%.1f tok/s, P99 latency=%.1fms",
            backend, results.throughput().getPeakTokensPerSecond(),
            results.latency().p99LatencyMs());

        return results;
    }

    /**
     * Runs throughput benchmark (tokens/sec at various batch sizes).
     */
    public ThroughputResult runThroughputBenchmark() {
        LOG.infof("Running throughput benchmark: backend=%s", backend);

        Map<Integer, ThroughputMetrics> results = new LinkedHashMap<>();

        for (int batchSize : batchSizes) {
            // Warmup
            for (int i = 0; i < warmupIterations; i++) {
                executeBatch(batchSize);
            }

            // Benchmark
            long totalTokens = 0;
            long totalTimeMs = 0;

            for (int i = 0; i < benchmarkIterations; i++) {
                Instant start = Instant.now();
                int tokens = executeBatch(batchSize);
                long elapsedMs = Duration.between(start, Instant.now()).toMillis();

                totalTokens += tokens;
                totalTimeMs += elapsedMs;
            }

            double tokensPerSec = (totalTimeMs > 0) ?
                (totalTokens * 1000.0 / totalTimeMs) : 0;
            double avgLatencyMs = totalTimeMs / (double) benchmarkIterations;

            results.put(batchSize, new ThroughputMetrics(
                batchSize, totalTokens, totalTimeMs, tokensPerSec, avgLatencyMs));

            LOG.debugf("Batch size %d: %.1f tok/s, avg %.1fms",
                (Object) batchSize, tokensPerSec, avgLatencyMs);
        }

        // Find optimal batch size (highest throughput)
        int optimalBatchSize = results.entrySet().stream()
            .max(Comparator.comparingDouble(e -> e.getValue().tokensPerSecond()))
            .map(Map.Entry::getKey)
            .orElse(1);

        return new ThroughputResult(results, optimalBatchSize);
    }

    /**
     * Runs latency benchmark (P50, P95, P99, TTFT).
     */
    public LatencyResult runLatencyBenchmark() {
        LOG.infof("Running latency benchmark: backend=%s", backend);

        List<Long> latencies = new ArrayList<>();
        List<Long> ttfts = new ArrayList<>();

        // Warmup
        for (int i = 0; i < warmupIterations; i++) {
            executeSingleRequest();
        }

        // Benchmark
        for (int i = 0; i < benchmarkIterations; i++) {
            Instant start = Instant.now();
            long ttftMs = executeSingleRequestWithTTFT();
            long totalMs = Duration.between(start, Instant.now()).toMillis();

            latencies.add(totalMs);
            ttfts.add(ttftMs);
        }

        Collections.sort(latencies);
        Collections.sort(ttfts);

        return new LatencyResult(
            latencies,
            ttfts,
            getPercentile(latencies, 50),
            getPercentile(latencies, 95),
            getPercentile(latencies, 99),
            getPercentile(ttfts, 50),
            getPercentile(ttfts, 95)
        );
    }

    /**
     * Runs concurrency benchmark (max concurrent requests).
     */
    public ConcurrencyResult runConcurrencyBenchmark() {
        LOG.infof("Running concurrency benchmark: backend=%s", backend);

        int concurrent = 1;
        int maxConcurrent = 1;
        int consecutiveFailures = 0;

        while (consecutiveFailures < 3) {
            try {
                boolean success = executeConcurrentRequests(concurrent);
                if (success) {
                    maxConcurrent = concurrent;
                    consecutiveFailures = 0;
                    concurrent *= 2;  // Exponential increase
                } else {
                    consecutiveFailures++;
                    concurrent = (concurrent + maxConcurrent) / 2;  // Binary search
                }
            } catch (Exception e) {
                consecutiveFailures++;
                concurrent = (concurrent + maxConcurrent) / 2;
            }

            if (concurrent <= maxConcurrent) break;
        }

        return new ConcurrencyResult(maxConcurrent, backend);
    }

    /**
     * Runs memory benchmark (VRAM usage).
     */
    public MemoryResult runMemoryBenchmark() {
        LOG.infof("Running memory benchmark: backend=%s", backend);

        // In production: query GPU memory usage
        // For now: estimate based on model size
        long modelSizeBytes = estimateModelSize();
        long kvCachePerRequest = estimateKVCacheSize();
        long maxConcurrent = modelSizeBytes > 0 ?
            (getAvailableMemory() - modelSizeBytes) / kvCachePerRequest : 0;

        return new MemoryResult(
            modelSizeBytes,
            kvCachePerRequest,
            getAvailableMemory(),
            maxConcurrent,
            backend
        );
    }

    /**
     * Runs context length scaling benchmark.
     */
    public ContextScalingResult runContextBenchmark() {
        LOG.infof("Running context scaling benchmark: backend=%s", backend);

        Map<Integer, ContextMetrics> results = new LinkedHashMap<>();

        for (int contextLen : contextLengths) {
            Instant start = Instant.now();
            int tokens = executeWithContextLength(contextLen);
            long elapsedMs = Duration.between(start, Instant.now()).toMillis();

            double tokensPerSec = elapsedMs > 0 ? (tokens * 1000.0 / elapsedMs) : 0;
            long memoryUsed = estimateKVCacheForContext(contextLen);

            results.put(contextLen, new ContextMetrics(
                contextLen, tokens, elapsedMs, tokensPerSec, memoryUsed));
        }

        return new ContextScalingResult(results, backend);
    }

    // ── Internal Execution Methods ────────────────────────────────────

    private int executeBatch(int batchSize) {
        // In production: send batch request to backend
        // Return total tokens generated
        return batchSize * maxTokens;
    }

    private long executeSingleRequestWithTTFT() {
        // In production: send single request, measure TTFT
        // Return TTFT in milliseconds
        return 50 + (long) (Math.random() * 100);  // Placeholder
    }

    private void executeSingleRequest() {
        // In production: send single request
    }

    private boolean executeConcurrentRequests(int concurrent) {
        // In production: send concurrent requests
        // Return true if all succeed
        return concurrent <= 100;  // Placeholder
    }

    private int executeWithContextLength(int contextLen) {
        // In production: send request with specified context length
        return maxTokens;
    }

    private long estimateModelSize() {
        // Estimate model size in bytes based on model name
        if (model.contains("70b")) return 70L * 1024 * 1024 * 1024 * 2;  // 70B × 2 bytes (FP16)
        if (model.contains("8b")) return 8L * 1024 * 1024 * 1024 * 2;
        if (model.contains("7b")) return 7L * 1024 * 1024 * 1024 * 2;
        return 140L * 1024 * 1024 * 1024;  // Default: 140B
    }

    private long estimateKVCacheSize() {
        // Estimate KV cache per request (depends on model architecture)
        return 280L * 1024 * 1024;  // ~280MB for 128 tokens (70B model)
    }

    private long getAvailableMemory() {
        // In production: query actual GPU memory
        return 80L * 1024 * 1024 * 1024;  // 80GB (A100)
    }

    private long estimateKVCacheForContext(int contextLen) {
        // Scale KV cache with context length
        return estimateKVCacheSize() * contextLen / 128L;
    }

    // ── Utility Methods ───────────────────────────────────────────────

    private long getPercentile(List<Long> sortedValues, double percentile) {
        if (sortedValues.isEmpty()) return 0;
        int index = (int) Math.ceil(percentile / 100.0 * sortedValues.size()) - 1;
        return sortedValues.get(Math.max(0, Math.min(index, sortedValues.size() - 1)));
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Backend platforms for comparison.
     */
    public enum BenchmarkBackend {
        GOLLEK("Gollek", "JVM, TurboQuant 3-bit"),
        VLLM("vLLM", "Python, FP8 KV cache"),
        TGI("TGI", "Python, FP8 KV cache"),
        ONNX_RUNTIME("ONNX Runtime", "C++, FP16"),
        TRITON("Triton", "C++, FP16");

        private final String displayName;
        private final String description;

        BenchmarkBackend(String displayName, String description) {
            this.displayName = displayName;
            this.description = description;
        }

        public String getDisplayName() { return displayName; }
        public String getDescription() { return description; }
    }

    /**
     * Throughput benchmark result.
     */
    public record ThroughputResult(
        Map<Integer, ThroughputMetrics> perBatchSize,
        int optimalBatchSize
    ) {
        public double getPeakTokensPerSecond() {
            return perBatchSize.values().stream()
                .mapToDouble(ThroughputMetrics::tokensPerSecond)
                .max().orElse(0);
        }
    }

    /**
     * Throughput metrics for a single batch size.
     */
    public record ThroughputMetrics(
        int batchSize,
        long totalTokens,
        long totalTimeMs,
        double tokensPerSecond,
        double avgLatencyMs
    ) {}

    /**
     * Latency benchmark result.
     */
    public record LatencyResult(
        List<Long> allLatencies,
        List<Long> allTTFTs,
        long p50LatencyMs,
        long p95LatencyMs,
        long p99LatencyMs,
        long p50TTFTMs,
        long p95TTFTMs
    ) {}

    /**
     * Concurrency benchmark result.
     */
    public record ConcurrencyResult(
        int maxConcurrentRequests,
        BenchmarkBackend backend
    ) {}

    /**
     * Memory benchmark result.
     */
    public record MemoryResult(
        long modelSizeBytes,
        long kvCachePerRequest,
        long totalAvailableMemory,
        long maxConcurrentRequests,
        BenchmarkBackend backend
    ) {
        public double modelSizeGB() { return modelSizeBytes / (1024.0 * 1024 * 1024); }
        public double totalMemoryGB() { return totalAvailableMemory / (1024.0 * 1024 * 1024); }
    }

    /**
     * Context length scaling result.
     */
    public record ContextScalingResult(
        Map<Integer, ContextMetrics> perContextLength,
        BenchmarkBackend backend
    ) {}

    /**
     * Context length metrics.
     */
    public record ContextMetrics(
        int contextLength,
        int tokensGenerated,
        long latencyMs,
        double tokensPerSecond,
        long memoryUsed
    ) {}

    /**
     * Complete benchmark results.
     */
    public record BenchmarkResults(
        BenchmarkBackend backend,
        String model,
        ThroughputResult throughput,
        LatencyResult latency,
        ConcurrencyResult concurrency,
        MemoryResult memory,
        ContextScalingResult contextScaling,
        Instant timestamp
    ) {
        public static Builder builder(BenchmarkBackend backend, String model) {
            return new Builder(backend, model);
        }

        public static class Builder {
            private final BenchmarkBackend backend;
            private final String model;
            private ThroughputResult throughput;
            private LatencyResult latency;
            private ConcurrencyResult concurrency;
            private MemoryResult memory;
            private ContextScalingResult contextScaling;

            private Builder(BenchmarkBackend backend, String model) {
                this.backend = backend;
                this.model = model;
            }

            public Builder throughput(ThroughputResult result) { this.throughput = result; return this; }
            public Builder latency(LatencyResult result) { this.latency = result; return this; }
            public Builder concurrency(ConcurrencyResult result) { this.concurrency = result; return this; }
            public Builder memory(MemoryResult result) { this.memory = result; return this; }
            public Builder contextScaling(ContextScalingResult result) { this.contextScaling = result; return this; }

            public BenchmarkResults build() {
                return new BenchmarkResults(backend, model, throughput, latency,
                    concurrency, memory, contextScaling, Instant.now());
            }
        }
    }

    /**
     * Benchmark comparison report.
     */
    public static final class BenchmarkReport {
        private final List<BenchmarkResults> results;

        private BenchmarkReport(List<BenchmarkResults> results) {
            this.results = List.copyOf(results);
        }

        /**
         * Creates a comparison report from multiple benchmark results.
         */
        public static BenchmarkReport compare(List<BenchmarkResults> results) {
            return new BenchmarkReport(results);
        }

        /**
         * Prints comparison table to stdout.
         */
        public void printTable() {
            System.out.println("\n=== Benchmark Comparison ===\n");

            // Header
            System.out.printf("%-20s", "Metric");
            for (BenchmarkResults r : results) {
                System.out.printf(" | %-15s", r.backend().getDisplayName());
            }
            System.out.println();
            System.out.println("-".repeat(20 + 18 * results.size()));

            // Throughput
            System.out.printf("%-20s", "Peak Throughput");
            for (BenchmarkResults r : results) {
                System.out.printf(" | %11.1f tok/s", r.throughput().getPeakTokensPerSecond());
            }
            System.out.println();

            // Latency P99
            System.out.printf("%-20s", "P99 Latency");
            for (BenchmarkResults r : results) {
                System.out.printf(" | %13dms", r.latency().p99LatencyMs());
            }
            System.out.println();

            // TTFT P50
            System.out.printf("%-20s", "P50 TTFT");
            for (BenchmarkResults r : results) {
                System.out.printf(" | %13dms", r.latency().p50TTFTMs());
            }
            System.out.println();

            // Max Concurrent
            System.out.printf("%-20s", "Max Concurrent");
            for (BenchmarkResults r : results) {
                System.out.printf(" | %14d", r.concurrency().maxConcurrentRequests());
            }
            System.out.println();

            // Max Context
            System.out.printf("%-20s", "Max Context");
            for (BenchmarkResults r : results) {
                int maxContext = r.contextScaling().perContextLength().keySet().stream()
                    .max(Integer::compareTo).orElse(0);
                System.out.printf(" | %12d", maxContext);
            }
            System.out.println();

            System.out.println();
        }
    }

    /**
     * Configuration for BenchmarkRunner.
     */
    private static final class Config {
        BenchmarkBackend backend = BenchmarkBackend.GOLLEK;
        String model = "llama-3-70b";
        int warmupIterations = 5;
        int benchmarkIterations = 20;
        List<Integer> batchSizes = List.of(1, 4, 16, 64, 128);
        List<Integer> contextLengths = List.of(128, 512, 2048, 4096);
        int maxTokens = 256;
        Duration requestTimeout = Duration.ofSeconds(30);
    }

    /**
     * Builder for BenchmarkRunner.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        public Builder backend(BenchmarkBackend backend) {
            config.backend = backend;
            return this;
        }

        public Builder model(String model) {
            config.model = model;
            return this;
        }

        public Builder warmupIterations(int iterations) {
            config.warmupIterations = iterations;
            return this;
        }

        public Builder benchmarkIterations(int iterations) {
            config.benchmarkIterations = iterations;
            return this;
        }

        public Builder batchSizes(List<Integer> sizes) {
            config.batchSizes = sizes;
            return this;
        }

        public Builder contextLengths(List<Integer> lengths) {
            config.contextLengths = lengths;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            config.maxTokens = maxTokens;
            return this;
        }

        public Builder requestTimeout(Duration timeout) {
            config.requestTimeout = timeout;
            return this;
        }

        public BenchmarkRunner build() {
            return new BenchmarkRunner(config);
        }
    }
}
