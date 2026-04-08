///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek LiteRT Performance Benchmark Example
//
// DEPENDENCY RESOLUTION:
// The SDK modules must be built and installed to local Maven repository first:
//   cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests
//
// REPOS mavenLocal
// DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-spi-inference:0.1.0-SNAPSHOT
// JAVA 21+
//
// Usage:
//   jbang BenchmarkExample.java                                    # Demo mode
//   jbang -Dmodel=/path/to/model.litert BenchmarkExample.java
//   jbang -Dmodel=model.litert -Diterations=100 BenchmarkExample.java
//   jbang -Dmodel=model.litert -Dbatch=8 -Dwarmup=10 BenchmarkExample.java
//   jbang -Dmodel=model.litert -Ddelegate=GPU -Dthreads=8 BenchmarkExample.java

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.LiteRTMetrics;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.sdk.litert.model.LiteRTModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * LiteRT Performance Benchmark Example
 *
 * Comprehensive benchmarking tool for measuring LiteRT inference performance:
 * - Latency benchmarks (single & batch)
 * - Throughput benchmarks
 * - Memory usage tracking
 * - Hardware delegate comparison
 * - Statistical analysis (mean, median, percentiles)
 *
 * Outputs detailed performance metrics and can generate CSV reports.
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class BenchmarkExample {

    // Benchmark results storage
    static class BenchmarkResult {
        String testName;
        long[] latencies;
        double throughput;
        long peakMemoryBytes;
        long currentMemoryBytes;

        BenchmarkResult(String testName, long[] latencies, double throughput,
                       long peakMemoryBytes, long currentMemoryBytes) {
            this.testName = testName;
            this.latencies = latencies;
            this.throughput = throughput;
            this.peakMemoryBytes = peakMemoryBytes;
            this.currentMemoryBytes = currentMemoryBytes;
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║          Gollek LiteRT Performance Benchmark             ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // Parse arguments
        String modelPath = System.getProperty("model", null);
        int iterations = Integer.parseInt(System.getProperty("iterations", "50"));
        int batchSize = Integer.parseInt(System.getProperty("batch", "1"));
        int warmupRuns = Integer.parseInt(System.getProperty("warmup", "10"));
        String delegateStr = System.getProperty("delegate", "AUTO");
        int numThreads = Integer.parseInt(System.getProperty("threads", "4"));
        boolean enableXnnpack = Boolean.parseBoolean(System.getProperty("xnnpack", "true"));
        String outputFile = System.getProperty("output", null);

        LiteRTConfig.Delegate delegate = LiteRTConfig.Delegate.valueOf(delegateStr.toUpperCase());

        if (modelPath == null || modelPath.isEmpty()) {
            System.out.println("Running in demo mode (no model provided)");
            System.out.println("To benchmark a real model:");
            System.out.println("  jbang -Dmodel=/path/to/model.litert BenchmarkExample.java");
            System.out.println();
            runDemoBenchmark();
            return;
        }

        // Validate model file
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            System.err.println("❌ Error: Model file not found: " + modelPath);
            System.exit(1);
        }

        System.out.println("Benchmark Configuration:");
        System.out.println("  Model:         " + modelPath);
        System.out.println("  Iterations:    " + iterations);
        System.out.println("  Batch Size:    " + batchSize);
        System.out.println("  Warmup Runs:   " + warmupRuns);
        System.out.println("  Delegate:      " + delegate);
        System.out.println("  Threads:       " + numThreads);
        System.out.println("  XNNPACK:       " + enableXnnpack);
        System.out.println();

        // Create SDK
        LiteRTConfig config = LiteRTConfig.builder()
                .numThreads(numThreads)
                .delegate(delegate)
                .enableXnnpack(enableXnnpack)
                .useMemoryPool(true)
                .build();

        List<BenchmarkResult> results = new ArrayList<>();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            // Load model
            System.out.println("Loading model...");
            sdk.loadModel("benchmark", Path.of(modelPath));
            LiteRTModelInfo modelInfo = sdk.getModelInfo("benchmark");
            System.out.println("✓ Model loaded: " + formatBytes(modelInfo.getModelSizeBytes()));
            System.out.println();

            // Warmup
            System.out.println("Running " + warmupRuns + " warmup iterations...");
            runWarmup(sdk, warmupRuns);
            System.out.println();

            // Benchmark 1: Single inference latency
            System.out.println("═".repeat(60));
            System.out.println("Benchmark 1: Single Inference Latency");
            System.out.println("═".repeat(60));
            BenchmarkResult singleResult = benchmarkSingleLatency(sdk, iterations);
            results.add(singleResult);
            printResult(singleResult);
            System.out.println();

            // Benchmark 2: Batch inference throughput
            if (batchSize > 1) {
                System.out.println("═".repeat(60));
                System.out.println("Benchmark 2: Batch Inference (batch=" + batchSize + ")");
                System.out.println("═".repeat(60));
                BenchmarkResult batchResult = benchmarkBatchThroughput(sdk, iterations, batchSize);
                results.add(batchResult);
                printResult(batchResult);
                System.out.println();
            }

            // Benchmark 3: Async inference concurrency
            System.out.println("═".repeat(60));
            System.out.println("Benchmark 3: Async Inference Concurrency");
            System.out.println("═".repeat(60));
            BenchmarkResult asyncResult = benchmarkAsyncConcurrency(sdk, iterations);
            results.add(asyncResult);
            printResult(asyncResult);
            System.out.println();

            // Benchmark 4: Sustained load
            System.out.println("═".repeat(60));
            System.out.println("Benchmark 4: Sustained Load (60 seconds)");
            System.out.println("═".repeat(60));
            BenchmarkResult sustainedResult = benchmarkSustainedLoad(sdk, 60);
            results.add(sustainedResult);
            printResult(sustainedResult);
            System.out.println();

            // Display final metrics
            System.out.println("═".repeat(60));
            System.out.println("Final Performance Metrics");
            System.out.println("═".repeat(60));
            displayMetrics(sdk);
            System.out.println();

            // Generate CSV report if requested
            if (outputFile != null) {
                generateCsvReport(outputFile, results);
                System.out.println("CSV report saved to: " + outputFile);
                System.out.println();
            }

            // Summary
            printSummary(results);
        }

        System.out.println();
        System.out.println("✓ Benchmark completed successfully");
    }

    /**
     * Run warmup iterations to initialize caches.
     */
    private static void runWarmup(LiteRTSdk sdk, int iterations) throws Exception {
        byte[] inputData = generateRandomInput(224 * 224 * 3 * 4);

        for (int i = 0; i < iterations; i++) {
            InferenceRequest request = InferenceRequest.builder()
                    .model("benchmark")
                    .inputData(inputData)
                    .build();
            sdk.infer(request);
        }
    }

    /**
     * Benchmark single inference latency.
     */
    private static BenchmarkResult benchmarkSingleLatency(LiteRTSdk sdk, int iterations) throws Exception {
        byte[] inputData = generateRandomInput(224 * 224 * 3 * 4);
        long[] latencies = new long[iterations];

        System.out.println("Running " + iterations + " single inference iterations...");

        for (int i = 0; i < iterations; i++) {
            InferenceRequest request = InferenceRequest.builder()
                    .model("benchmark")
                    .inputData(inputData)
                    .build();

            long start = System.nanoTime();
            sdk.infer(request);
            long elapsed = System.nanoTime() - start;
            latencies[i] = elapsed / 1_000_000; // Convert to ms
        }

        LiteRTMetrics metrics = sdk.getMetrics(null);
        double throughput = iterations * 1000.0 / calculateSum(latencies);

        return new BenchmarkResult("Single Inference", latencies, throughput,
                metrics.getPeakMemoryBytes(), metrics.getCurrentMemoryBytes());
    }

    /**
     * Benchmark batch inference throughput.
     */
    private static BenchmarkResult benchmarkBatchThroughput(LiteRTSdk sdk, int iterations, int batchSize) throws Exception {
        byte[] inputData = generateRandomInput(224 * 224 * 3 * 4);
        long[] latencies = new long[iterations];

        System.out.println("Running " + iterations + " batch inference iterations (batch=" + batchSize + ")...");

        for (int i = 0; i < iterations; i++) {
            List<InferenceRequest> batchRequests = IntStream.range(0, batchSize)
                    .mapToObj(j -> InferenceRequest.builder()
                            .model("benchmark")
                            .inputData(inputData)
                            .build())
                    .toList();

            long start = System.nanoTime();
            sdk.inferBatch(batchRequests);
            long elapsed = System.nanoTime() - start;
            latencies[i] = elapsed / 1_000_000;
        }

        LiteRTMetrics metrics = sdk.getMetrics(null);
        double throughput = (iterations * batchSize) * 1000.0 / calculateSum(latencies);

        return new BenchmarkResult("Batch Inference (batch=" + batchSize + ")", latencies, throughput,
                metrics.getPeakMemoryBytes(), metrics.getCurrentMemoryBytes());
    }

    /**
     * Benchmark async inference concurrency.
     */
    private static BenchmarkResult benchmarkAsyncConcurrency(LiteRTSdk sdk, int iterations) throws Exception {
        byte[] inputData = generateRandomInput(224 * 224 * 3 * 4);
        long[] latencies = new long[iterations];
        int concurrency = Math.min(8, iterations);

        System.out.println("Running " + iterations + " async inference requests (concurrency=" + concurrency + ")...");

        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        List<Future<Long>> futures = new ArrayList<>();

        for (int i = 0; i < iterations; i++) {
            final int index = i;
            Future<Long> future = executor.submit(() -> {
                InferenceRequest request = InferenceRequest.builder()
                        .model("benchmark")
                        .inputData(inputData)
                        .build();

                long start = System.nanoTime();
                sdk.inferAsync(request).get();
                return (System.nanoTime() - start) / 1_000_000;
            });
            futures.add(future);
        }

        // Collect results
        for (int i = 0; i < iterations; i++) {
            latencies[i] = futures.get(i).get();
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        LiteRTMetrics metrics = sdk.getMetrics(null);
        double throughput = iterations * 1000.0 / calculateSum(latencies);

        return new BenchmarkResult("Async Inference (concurrency=" + concurrency + ")", latencies, throughput,
                metrics.getPeakMemoryBytes(), metrics.getCurrentMemoryBytes());
    }

    /**
     * Benchmark sustained load over time.
     */
    private static BenchmarkResult benchmarkSustainedLoad(LiteRTSdk sdk, int durationSeconds) throws Exception {
        byte[] inputData = generateRandomInput(224 * 224 * 3 * 4);
        List<Long> latencies = new ArrayList<>();

        System.out.println("Running sustained load for " + durationSeconds + " seconds...");

        long startTime = System.currentTimeMillis();
        long endTime = startTime + (durationSeconds * 1000L);
        int completed = 0;

        while (System.currentTimeMillis() < endTime) {
            InferenceRequest request = InferenceRequest.builder()
                    .model("benchmark")
                    .inputData(inputData)
                    .build();

            long start = System.nanoTime();
            sdk.infer(request);
            long elapsed = System.nanoTime() - start;
            latencies.add(elapsed / 1_000_000);
            completed++;
        }

        long[] latencyArray = latencies.stream().mapToLong(Long::longValue).toArray();
        double throughput = completed * 1000.0 / calculateSum(latencyArray);

        LiteRTMetrics metrics = sdk.getMetrics(null);

        return new BenchmarkResult("Sustained Load (" + durationSeconds + "s)", latencyArray, throughput,
                metrics.getPeakMemoryBytes(), metrics.getCurrentMemoryBytes());
    }

    /**
     * Run demo benchmark with synthetic data.
     */
    private static void runDemoBenchmark() {
        System.out.println("Demo mode - simulating benchmark with synthetic data");
        System.out.println();

        int iterations = 50;
        long[] latencies = new long[iterations];
        Random random = new Random(42);

        System.out.println("Running " + iterations + " simulated iterations...");

        for (int i = 0; i < iterations; i++) {
            latencies[i] = 10 + (long) (random.nextGaussian() * 5 + 5);
            System.out.print("\r  Progress: " + (i + 1) + "/" + iterations);
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println();
        System.out.println();

        // Print simulated results
        System.out.println("═".repeat(60));
        System.out.println("Simulated Benchmark Results");
        System.out.println("═".repeat(60));
        System.out.println("  Mean Latency:    " + String.format("%.2f", calculateMean(latencies)) + " ms");
        System.out.println("  Median Latency:  " + String.format("%.2f", calculateMedian(latencies)) + " ms");
        System.out.println("  P50 Latency:     " + String.format("%.2f", calculatePercentile(latencies, 50)) + " ms");
        System.out.println("  P95 Latency:     " + String.format("%.2f", calculatePercentile(latencies, 95)) + " ms");
        System.out.println("  P99 Latency:     " + String.format("%.2f", calculatePercentile(latencies, 99)) + " ms");
        System.out.println("  Min Latency:     " + calculateMin(latencies) + " ms");
        System.out.println("  Max Latency:     " + calculateMax(latencies) + " ms");
        System.out.println("  Std Deviation:   " + String.format("%.2f", calculateStdDev(latencies)) + " ms");
        System.out.println("  Throughput:      " + String.format("%.2f", iterations * 1000.0 / calculateSum(latencies)) + " inf/sec");
        System.out.println();
        System.out.println("To run with a real model:");
        System.out.println("  jbang -Dmodel=/path/to/model.litert BenchmarkExample.java");
    }

    /**
     * Print benchmark result.
     */
    private static void printResult(BenchmarkResult result) {
        System.out.println("  Test:            " + result.testName);
        System.out.println("  Mean Latency:    " + String.format("%.2f", calculateMean(result.latencies)) + " ms");
        System.out.println("  Median Latency:  " + String.format("%.2f", calculateMedian(result.latencies)) + " ms");
        System.out.println("  P50 Latency:     " + String.format("%.2f", calculatePercentile(result.latencies, 50)) + " ms");
        System.out.println("  P95 Latency:     " + String.format("%.2f", calculatePercentile(result.latencies, 95)) + " ms");
        System.out.println("  P99 Latency:     " + String.format("%.2f", calculatePercentile(result.latencies, 99)) + " ms");
        System.out.println("  Min Latency:     " + calculateMin(result.latencies) + " ms");
        System.out.println("  Max Latency:     " + calculateMax(result.latencies) + " ms");
        System.out.println("  Std Deviation:   " + String.format("%.2f", calculateStdDev(result.latencies)) + " ms");
        System.out.println("  Throughput:      " + String.format("%.2f", result.throughput) + " inf/sec");
        System.out.println("  Peak Memory:     " + formatBytes(result.peakMemoryBytes));
    }

    /**
     * Print benchmark summary.
     */
    private static void printSummary(List<BenchmarkResult> results) {
        System.out.println("═".repeat(60));
        System.out.println("Benchmark Summary");
        System.out.println("═".repeat(60));

        for (BenchmarkResult result : results) {
            System.out.printf("  %-40s %8.2f ms (mean)  %8.2f inf/sec%n",
                    result.testName, calculateMean(result.latencies), result.throughput);
        }
    }

    /**
     * Generate CSV report.
     */
    private static void generateCsvReport(String outputFile, List<BenchmarkResult> results) throws Exception {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(outputFile))) {
            writer.println("Test,Mean_ms,Median_ms,P50_ms,P95_ms,P99_ms,Min_ms,Max_ms,StdDev_ms,Throughput_inf_sec,Peak_Memory_B");

            for (BenchmarkResult result : results) {
                writer.printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f,%.2f,%d%n",
                        result.testName,
                        calculateMean(result.latencies),
                        calculateMedian(result.latencies),
                        calculatePercentile(result.latencies, 50),
                        calculatePercentile(result.latencies, 95),
                        calculatePercentile(result.latencies, 99),
                        calculateMin(result.latencies),
                        calculateMax(result.latencies),
                        calculateStdDev(result.latencies),
                        result.throughput,
                        result.peakMemoryBytes);
            }
        }
    }

    /**
     * Display performance metrics from SDK.
     */
    private static void displayMetrics(LiteRTSdk sdk) {
        try {
            LiteRTMetrics metrics = sdk.getMetrics(null);
            System.out.println("  Total inferences:    " + metrics.getTotalInferences());
            System.out.println("  Failed inferences:   " + metrics.getFailedInferences());
            System.out.println("  Avg latency:         " + String.format("%.2f", metrics.getAvgLatencyMs()) + " ms");
            System.out.println("  P50 latency:         " + String.format("%.2f", metrics.getP50LatencyMs()) + " ms");
            System.out.println("  P95 latency:         " + String.format("%.2f", metrics.getP95LatencyMs()) + " ms");
            System.out.println("  P99 latency:         " + String.format("%.2f", metrics.getP99LatencyMs()) + " ms");
            System.out.println("  Peak memory:         " + formatBytes(metrics.getPeakMemoryBytes()));
            System.out.println("  Current memory:      " + formatBytes(metrics.getCurrentMemoryBytes()));
            System.out.println("  Active delegate:     " + metrics.getActiveDelegate());
        } catch (Exception e) {
            System.out.println("  (Metrics not available)");
        }
    }

    // Statistical helper methods

    private static double calculateMean(long[] values) {
        return calculateSum(values) / (double) values.length;
    }

    private static long calculateSum(long[] values) {
        long sum = 0;
        for (long v : values) sum += v;
        return sum;
    }

    private static double calculateMedian(long[] values) {
        long[] sorted = values.clone();
        java.util.Arrays.sort(sorted);
        int mid = sorted.length / 2;
        if (sorted.length % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        } else {
            return sorted[mid];
        }
    }

    private static double calculatePercentile(long[] values, double percentile) {
        long[] sorted = values.clone();
        java.util.Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    private static long calculateMin(long[] values) {
        long min = Long.MAX_VALUE;
        for (long v : values) min = Math.min(min, v);
        return min;
    }

    private static long calculateMax(long[] values) {
        long max = Long.MIN_VALUE;
        for (long v : values) max = Math.max(max, v);
        return max;
    }

    private static double calculateStdDev(long[] values) {
        double mean = calculateMean(values);
        double sumSquaredDiff = 0;
        for (long v : values) {
            double diff = v - mean;
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff / values.length);
    }

    private static byte[] generateRandomInput(int size) {
        byte[] input = new byte[size];
        new Random().nextBytes(input);
        return input;
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
