/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.gollek.model.evaluation;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelManifest;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Service for running standardized evaluation harnesses against registered models.
 *
 * <h3>Supported Benchmarks</h3>
 * <ul>
 *   <li><b>MMLU</b>   – Massive Multitask Language Understanding (accuracy %)</li>
 *   <li><b>HumanEval</b> – Code generation pass@k score</li>
 *   <li><b>HellaSwag</b> – Commonsense NLI accuracy</li>
 *   <li><b>TruthfulQA</b> – Truthfulness percentage</li>
 *   <li><b>MT-Bench</b> – Multi-turn conversation quality (1–10 scale)</li>
 *   <li><b>Custom</b> – User-defined benchmark via {@link BenchmarkTask}</li>
 * </ul>
 *
 * <h3>Storage</h3>
 * Evaluation scores are stored in the {@link ModelManifest} {@code metadata} map
 * under the key {@code "evaluationScores"} as a
 * {@code Map<String, EvaluationScore>} keyed by benchmark name.
 *
 * <h3>Usage</h3>
 * <pre>
 * evaluationService.evaluate(manifest, List.of(Benchmark.MMLU, Benchmark.HUMAN_EVAL));
 * EvaluationReport report = evaluationService.getReport(manifest.modelId());
 * </pre>
 */
@ApplicationScoped
public class ModelEvaluationService {

    private static final Logger LOG = Logger.getLogger(ModelEvaluationService.class.getName());

    /** Metadata key used inside {@link ModelManifest#metadata()} to store scores. */
    public static final String METADATA_KEY_SCORES = "evaluationScores";

    /** In-memory store: modelId → list of evaluation reports */
    private final Map<String, List<EvaluationReport>> reportStore = new ConcurrentHashMap<>();

    /** Pluggable benchmark runners */
    private final Map<Benchmark, BenchmarkTask> benchmarkTasks = new ConcurrentHashMap<>();

    // -----------------------------------------------------------------------
    // Benchmark enum
    // -----------------------------------------------------------------------

    public enum Benchmark {
        MMLU("Massive Multitask Language Understanding", "accuracy", 100.0),
        HUMAN_EVAL("HumanEval pass@1", "pass@1", 100.0),
        HELLASWAG("HellaSwag accuracy", "accuracy", 100.0),
        TRUTHFUL_QA("TruthfulQA", "truthfulness_%", 100.0),
        MT_BENCH("MT-Bench", "score", 10.0),
        GSME8K("GSM8K Math", "accuracy", 100.0),
        CUSTOM("Custom", "custom", Double.MAX_VALUE);

        public final String displayName;
        public final String metricUnit;
        public final double maxScore;

        Benchmark(String displayName, String metricUnit, double maxScore) {
            this.displayName = displayName;
            this.metricUnit  = metricUnit;
            this.maxScore    = maxScore;
        }
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /**
     * Runs one or more benchmarks against the given model and returns a report.
     *
     * @param manifest   the model to evaluate
     * @param benchmarks list of benchmarks to run
     * @return evaluation report with all scores
     */
    public EvaluationReport evaluate(ModelManifest manifest, List<Benchmark> benchmarks) {
        LOG.info("[Eval] Starting evaluation | model=" + manifest.modelId() +
                 " benchmarks=" + benchmarks);

        Map<Benchmark, EvaluationScore> scores = new LinkedHashMap<>();
        for (Benchmark benchmark : benchmarks) {
            BenchmarkTask task = benchmarkTasks.get(benchmark);
            if (task == null) {
                LOG.warning("[Eval] No task registered for benchmark=" + benchmark + "; using stub");
                task = stubTask(benchmark);
            }
            try {
                EvaluationScore score = task.run(manifest);
                scores.put(benchmark, score);
                LOG.info(String.format("[Eval] %s | model=%s score=%.2f %s",
                    benchmark.displayName, manifest.modelId(),
                    score.value(), benchmark.metricUnit));
            } catch (Exception e) {
                LOG.warning("[Eval] Benchmark " + benchmark + " failed: " + e.getMessage());
                scores.put(benchmark, EvaluationScore.error(e.getMessage()));
            }
        }

        EvaluationReport report = new EvaluationReport(
            manifest.modelId(), manifest.version(),
            Instant.now(), scores,
            computeOverallScore(scores)
        );
        reportStore.computeIfAbsent(manifest.modelId(), k -> new ArrayList<>()).add(report);
        return report;
    }

    /**
     * Returns the latest evaluation report for a model, or empty if none exists.
     */
    public Optional<EvaluationReport> getLatestReport(String modelId) {
        List<EvaluationReport> reports = reportStore.get(modelId);
        if (reports == null || reports.isEmpty()) return Optional.empty();
        return Optional.of(reports.get(reports.size() - 1));
    }

    /**
     * Returns all historical evaluation reports for a model.
     */
    public List<EvaluationReport> getHistory(String modelId) {
        return reportStore.getOrDefault(modelId, List.of());
    }

    /**
     * Registers a custom benchmark task for a given benchmark type.
     */
    public void registerBenchmarkTask(Benchmark benchmark, BenchmarkTask task) {
        benchmarkTasks.put(benchmark, task);
    }

    // -----------------------------------------------------------------------
    // Domain objects
    // -----------------------------------------------------------------------

    /**
     * The result of running a single benchmark.
     *
     * @param value      numeric score
     * @param error      error message if the benchmark failed, otherwise null
     * @param sampleSize number of evaluation examples used
     * @param metadata   additional benchmark-specific metadata
     */
    public record EvaluationScore(
        double value,
        String error,
        int sampleSize,
        Map<String, Object> metadata
    ) {
        public static EvaluationScore of(double value, int sampleSize) {
            return new EvaluationScore(value, null, sampleSize, Map.of());
        }
        public static EvaluationScore error(String msg) {
            return new EvaluationScore(Double.NaN, msg, 0, Map.of());
        }
        public boolean isSuccess() { return error == null; }
    }

    /**
     * A complete evaluation report for one model run.
     *
     * @param modelId       model being evaluated
     * @param modelVersion  version of the model
     * @param evaluatedAt   timestamp of evaluation
     * @param scores        per-benchmark scores
     * @param overallScore  composite normalised score (0–1)
     */
    public record EvaluationReport(
        String modelId,
        String modelVersion,
        Instant evaluatedAt,
        Map<Benchmark, EvaluationScore> scores,
        double overallScore
    ) {}

    /** SPI: implement to run a specific benchmark against a model. */
    @FunctionalInterface
    public interface BenchmarkTask {
        EvaluationScore run(ModelManifest manifest) throws Exception;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private double computeOverallScore(Map<Benchmark, EvaluationScore> scores) {
        return scores.values().stream()
            .filter(EvaluationScore::isSuccess)
            .mapToDouble(EvaluationScore::value)
            .average()
            .orElse(Double.NaN);
    }

    /**
     * Stub task that returns a placeholder score when no real task is registered.
     * Useful for unit testing and initial integration.
     */
    private BenchmarkTask stubTask(Benchmark benchmark) {
        return manifest -> {
            LOG.warning("[Eval] Stub score returned for benchmark=" + benchmark.displayName);
            return EvaluationScore.of(-1.0, 0); // sentinel: not yet measured
        };
    }
}
