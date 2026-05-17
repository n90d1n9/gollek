package tech.kayys.gollek.ml.export;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

import java.util.ArrayList;
import java.util.List;

/**
 * Model benchmark for measuring inference performance.
 */
public class Benchmark {

    private final Object model;

    public Benchmark(Object model) {
        this.model = model;
    }

    public BenchmarkResult run(long[] inputShape, int iterations, int warmup) {
        List<Double> latencies = new ArrayList<>();

        for (int i = 0; i < warmup; i++) {
            runInference(inputShape);
        }

        long totalStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            runInference(inputShape);
            latencies.add((System.nanoTime() - start) / 1_000_000.0);
        }
        long totalElapsed = System.nanoTime() - totalStart;

        double avgLatency = latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double p50 = getPercentile(latencies, 50);
        double p95 = getPercentile(latencies, 95);
        double p99 = getPercentile(latencies, 99);
        double throughput = iterations * 1e9 / totalElapsed;

        return new BenchmarkResult(avgLatency, p50, p95, p99, throughput, iterations, inputShape);
    }

    public BenchmarkResult run(long[] inputShape) {
        return run(inputShape, 100, 10);
    }

    private void runInference(long[] inputShape) {
        if (model instanceof NNModule module) {
            GradTensor input = GradTensor.randn(inputShape);
            module.eval();
            module.forward(input);
        }
    }

    private double getPercentile(List<Double> values, double percentile) {
        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compareTo);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, index));
    }

    public record BenchmarkResult(
            double avgLatencyMs,
            double p50LatencyMs,
            double p95LatencyMs,
            double p99LatencyMs,
            double throughput,
            int iterations,
            long[] inputShape) {
        @Override
        public String toString() {
            return String.format(
                    "BenchmarkResult{avg=%.2fms, p50=%.2fms, p95=%.2fms, p99=%.2fms, throughput=%.1f inf/s, iterations=%d}",
                    avgLatencyMs, p50LatencyMs, p95LatencyMs, p99LatencyMs, throughput, iterations);
        }
    }
}
