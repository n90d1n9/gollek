package tech.kayys.gollek.export;

/**
 * @deprecated Use {@link tech.kayys.gollek.ml.export.Benchmark} or
 *             {@link tech.kayys.gollek.ml.Gollek.Export#benchmark(Object)} instead.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class Benchmark {

    private final tech.kayys.gollek.ml.export.Benchmark delegate;

    public Benchmark(Object model) {
        this.delegate = new tech.kayys.gollek.ml.export.Benchmark(model);
    }

    public BenchmarkResult run(long[] inputShape, int iterations, int warmup) {
        var result = delegate.run(inputShape, iterations, warmup);
        return new BenchmarkResult(
                result.avgLatencyMs(),
                result.p50LatencyMs(),
                result.p95LatencyMs(),
                result.p99LatencyMs(),
                result.throughput(),
                result.iterations(),
                result.inputShape());
    }

    public BenchmarkResult run(long[] inputShape) {
        var result = delegate.run(inputShape);
        return new BenchmarkResult(
                result.avgLatencyMs(),
                result.p50LatencyMs(),
                result.p95LatencyMs(),
                result.p99LatencyMs(),
                result.throughput(),
                result.iterations(),
                result.inputShape());
    }

    public record BenchmarkResult(
            double avgLatencyMs,
            double p50LatencyMs,
            double p95LatencyMs,
            double p99LatencyMs,
            double throughput,
            int iterations,
            long[] inputShape) {
    }
}
