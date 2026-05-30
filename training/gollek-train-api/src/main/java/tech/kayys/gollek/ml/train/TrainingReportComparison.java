package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Compares two canonical training reports for experiment regression checks.
 */
public record TrainingReportComparison(
        TrainingReport baseline,
        TrainingReport candidate,
        Options options,
        List<MetricDelta> metrics,
        List<TrainingReportDiagnostics.Finding> findings) {
    public static final Options DEFAULT_OPTIONS = new Options(1.0e-8, 0.05, 1.25, 4.0, 2.0);

    public TrainingReportComparison {
        baseline = Objects.requireNonNull(baseline, "baseline must not be null");
        candidate = Objects.requireNonNull(candidate, "candidate must not be null");
        options = options == null ? DEFAULT_OPTIONS : options;
        metrics = metrics == null ? List.of() : List.copyOf(metrics);
        findings = findings == null ? List.of() : List.copyOf(findings);
    }

    public enum Direction {
        LOWER_IS_BETTER,
        HIGHER_IS_BETTER,
        NEUTRAL
    }

    public record Options(
            double lossRegressionTolerance,
            double generalizationGapRegressionTolerance,
            double durationRegressionRatio,
            double gradientL2NormRegressionRatio,
            double parameterUpdateToParameterL2RegressionRatio) {
        public Options(
                double lossRegressionTolerance,
                double generalizationGapRegressionTolerance,
                double durationRegressionRatio) {
            this(
                    lossRegressionTolerance,
                    generalizationGapRegressionTolerance,
                    durationRegressionRatio,
                    4.0,
                    2.0);
        }

        public Options {
            lossRegressionTolerance = nonNegativeOrDefault(lossRegressionTolerance, 1.0e-8);
            generalizationGapRegressionTolerance = nonNegativeOrDefault(generalizationGapRegressionTolerance, 0.05);
            durationRegressionRatio = durationRegressionRatio > 1.0 && Double.isFinite(durationRegressionRatio)
                    ? durationRegressionRatio
                    : 1.25;
            gradientL2NormRegressionRatio = ratioOrDefault(gradientL2NormRegressionRatio, 4.0);
            parameterUpdateToParameterL2RegressionRatio = ratioOrDefault(
                    parameterUpdateToParameterL2RegressionRatio,
                    2.0);
        }
    }

    public record MetricDelta(
            String name,
            Direction direction,
            OptionalDouble baselineValue,
            OptionalDouble candidateValue) {
        public MetricDelta {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            name = name.trim();
            direction = direction == null ? Direction.NEUTRAL : direction;
            baselineValue = baselineValue == null ? OptionalDouble.empty() : baselineValue;
            candidateValue = candidateValue == null ? OptionalDouble.empty() : candidateValue;
        }

        public boolean available() {
            return baselineValue.isPresent() && candidateValue.isPresent();
        }

        public OptionalDouble absoluteDelta() {
            if (!available()) {
                return OptionalDouble.empty();
            }
            return OptionalDouble.of(candidateValue.getAsDouble() - baselineValue.getAsDouble());
        }

        public OptionalDouble relativeDelta() {
            if (!available()) {
                return OptionalDouble.empty();
            }
            double baseline = baselineValue.getAsDouble();
            if (Math.abs(baseline) < 1.0e-12) {
                return OptionalDouble.empty();
            }
            return OptionalDouble.of(absoluteDelta().orElseThrow() / Math.abs(baseline));
        }

        public boolean improved(double tolerance) {
            OptionalDouble delta = absoluteDelta();
            if (delta.isEmpty()) {
                return false;
            }
            double resolvedTolerance = Math.max(0.0, tolerance);
            return switch (direction) {
                case LOWER_IS_BETTER -> delta.getAsDouble() < -resolvedTolerance;
                case HIGHER_IS_BETTER -> delta.getAsDouble() > resolvedTolerance;
                case NEUTRAL -> false;
            };
        }

        public boolean regressed(double tolerance) {
            OptionalDouble delta = absoluteDelta();
            if (delta.isEmpty()) {
                return false;
            }
            double resolvedTolerance = Math.max(0.0, tolerance);
            return switch (direction) {
                case LOWER_IS_BETTER -> delta.getAsDouble() > resolvedTolerance;
                case HIGHER_IS_BETTER -> delta.getAsDouble() < -resolvedTolerance;
                case NEUTRAL -> false;
            };
        }

        public Map<String, Object> toMap() {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", name);
            row.put("direction", direction.name());
            row.put("available", available());
            putOptional(row, "baseline", baselineValue);
            putOptional(row, "candidate", candidateValue);
            putOptional(row, "absoluteDelta", absoluteDelta());
            putOptional(row, "relativeDelta", relativeDelta());
            return Map.copyOf(row);
        }
    }

    public static TrainingReportComparison compare(Path baselineReportFile, Path candidateReportFile)
            throws IOException {
        return compare(TrainingReportReader.readReport(baselineReportFile), TrainingReportReader.readReport(candidateReportFile));
    }

    public static TrainingReportComparison compare(TrainingReport baseline, TrainingReport candidate) {
        return compare(baseline, candidate, DEFAULT_OPTIONS);
    }

    public static TrainingReportComparison compare(
            TrainingReport baseline,
            TrainingReport candidate,
            Options options) {
        Options resolvedOptions = options == null ? DEFAULT_OPTIONS : options;
        List<MetricDelta> metrics = metrics(baseline, candidate);
        List<TrainingReportDiagnostics.Finding> findings = findings(baseline, candidate, metrics, resolvedOptions);
        return new TrainingReportComparison(baseline, candidate, resolvedOptions, metrics, findings);
    }

    public Optional<MetricDelta> metric(String name) {
        if (name == null || name.isBlank()) {
            return Optional.empty();
        }
        return metrics.stream()
                .filter(metric -> metric.name().equals(name))
                .findFirst();
    }

    public boolean hasFindings() {
        return !findings.isEmpty();
    }

    public TrainingReportDiagnostics.Summary summary() {
        return TrainingReportDiagnostics.summarize(findings);
    }

    public TrainingReportDiagnostics.GateResult gate(TrainingReportDiagnostics.Severity maxAllowedSeverity) {
        return TrainingReportDiagnostics.gateFindings(findings, maxAllowedSeverity);
    }

    public List<MetricDelta> regressedMetrics() {
        return metrics.stream()
                .filter(metric -> isMetricRegressed(metric, options))
                .toList();
    }

    public List<MetricDelta> improvedMetrics() {
        return metrics.stream()
                .filter(metric -> metric.improved(toleranceFor(metric.name(), options)))
                .toList();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("baseline", reportSummary(baseline));
        map.put("candidate", reportSummary(candidate));
        map.put("metrics", metrics.stream().map(MetricDelta::toMap).toList());
        map.put("findings", TrainingReportDiagnostics.toMaps(findings));
        map.put("summary", summary().toMap());
        return Map.copyOf(map);
    }

    public TrainingReportComparisonExport export() {
        return TrainingReportComparisonExport.fromComparison(this);
    }

    private static List<MetricDelta> metrics(TrainingReport baseline, TrainingReport candidate) {
        Objects.requireNonNull(baseline, "baseline must not be null");
        Objects.requireNonNull(candidate, "candidate must not be null");
        List<MetricDelta> metrics = new ArrayList<>();
        metrics.add(metric(
                "bestValidationLoss",
                Direction.LOWER_IS_BETTER,
                baseline.bestValidationLoss(),
                candidate.bestValidationLoss()));
        metrics.add(metric(
                "latestValidationLoss",
                Direction.LOWER_IS_BETTER,
                baseline.latestValidationLoss(),
                candidate.latestValidationLoss()));
        metrics.add(metric(
                "latestTrainLoss",
                Direction.LOWER_IS_BETTER,
                baseline.latestTrainLoss(),
                candidate.latestTrainLoss()));
        metrics.add(metric(
                "latestGeneralizationGap",
                Direction.LOWER_IS_BETTER,
                baseline.latestGeneralizationGap(),
                candidate.latestGeneralizationGap()));
        metrics.add(metric(
                "epochCount",
                Direction.NEUTRAL,
                OptionalDouble.of(baseline.epochCount()),
                OptionalDouble.of(candidate.epochCount())));
        metrics.add(metric(
                "durationMs",
                Direction.LOWER_IS_BETTER,
                positiveDuration(baseline.durationMs()),
                positiveDuration(candidate.durationMs())));
        metrics.add(metric(
                "latestGradientL2Norm",
                Direction.NEUTRAL,
                baseline.latestGradientL2Norm(),
                candidate.latestGradientL2Norm()));
        metrics.add(metric(
                "latestParameterUpdateToParameterL2Ratio",
                Direction.NEUTRAL,
                baseline.latestParameterUpdateToParameterL2Ratio(),
                candidate.latestParameterUpdateToParameterL2Ratio()));
        return List.copyOf(metrics);
    }

    private static List<TrainingReportDiagnostics.Finding> findings(
            TrainingReport baseline,
            TrainingReport candidate,
            List<MetricDelta> metrics,
            Options options) {
        List<TrainingReportDiagnostics.Finding> findings = new ArrayList<>();
        addRegressionFinding(
                findings,
                metric(metrics, "bestValidationLoss"),
                options.lossRegressionTolerance(),
                "comparison.best_validation_loss.regressed",
                "Candidate best validation loss is higher than the baseline.");
        addRegressionFinding(
                findings,
                metric(metrics, "latestValidationLoss"),
                options.lossRegressionTolerance(),
                "comparison.latest_validation_loss.regressed",
                "Candidate latest validation loss is higher than the baseline.");
        addRegressionFinding(
                findings,
                metric(metrics, "latestTrainLoss"),
                options.lossRegressionTolerance(),
                "comparison.latest_train_loss.regressed",
                "Candidate latest training loss is higher than the baseline.");
        addRegressionFinding(
                findings,
                metric(metrics, "latestGeneralizationGap"),
                options.generalizationGapRegressionTolerance(),
                "comparison.generalization_gap.regressed",
                "Candidate validation-to-training loss gap is wider than the baseline.");
        addDurationRegression(findings, metric(metrics, "durationMs"), options);
        addRatioRegression(
                findings,
                metric(metrics, "latestGradientL2Norm"),
                options.gradientL2NormRegressionRatio(),
                "comparison.gradient_l2_norm.spiked",
                "Candidate latest gradient L2 norm is much larger than the baseline.");
        addRatioRegression(
                findings,
                metric(metrics, "latestParameterUpdateToParameterL2Ratio"),
                options.parameterUpdateToParameterL2RegressionRatio(),
                "comparison.parameter_update_ratio.regressed",
                "Candidate latest parameter-update-to-parameter ratio is much larger than the baseline.");
        addDiagnosticRegression(findings, baseline, candidate);
        return List.copyOf(findings);
    }

    private static void addRegressionFinding(
            List<TrainingReportDiagnostics.Finding> findings,
            Optional<MetricDelta> maybeMetric,
            double tolerance,
            String code,
            String message) {
        if (maybeMetric.isEmpty() || !maybeMetric.get().regressed(tolerance)) {
            return;
        }
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                code,
                message,
                evidence(maybeMetric.get(), tolerance)));
    }

    private static void addDurationRegression(
            List<TrainingReportDiagnostics.Finding> findings,
            Optional<MetricDelta> maybeMetric,
            Options options) {
        if (maybeMetric.isEmpty() || !maybeMetric.get().available()) {
            return;
        }
        MetricDelta metric = maybeMetric.get();
        double baseline = metric.baselineValue().orElseThrow();
        double candidate = metric.candidateValue().orElseThrow();
        if (baseline <= 0.0) {
            return;
        }
        double ratio = candidate / baseline;
        if (ratio <= options.durationRegressionRatio()) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>(evidence(metric, 0.0));
        evidence.put("durationRatio", ratio);
        evidence.put("maxAllowedDurationRatio", options.durationRegressionRatio());
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                "comparison.duration.regressed",
                "Candidate training duration is slower than the baseline threshold.",
                evidence));
    }

    private static void addRatioRegression(
            List<TrainingReportDiagnostics.Finding> findings,
            Optional<MetricDelta> maybeMetric,
            double maxAllowedRatio,
            String code,
            String message) {
        if (maybeMetric.isEmpty() || !maybeMetric.get().available()) {
            return;
        }
        MetricDelta metric = maybeMetric.get();
        double baseline = metric.baselineValue().orElseThrow();
        double candidate = metric.candidateValue().orElseThrow();
        if (!Double.isFinite(baseline)
                || !Double.isFinite(candidate)
                || baseline <= 0.0
                || candidate <= 0.0) {
            return;
        }
        double ratio = candidate / baseline;
        if (ratio <= maxAllowedRatio) {
            return;
        }
        Map<String, Object> evidence = new LinkedHashMap<>(evidence(metric, 0.0));
        evidence.put("ratio", ratio);
        evidence.put("maxAllowedRatio", maxAllowedRatio);
        findings.add(new TrainingReportDiagnostics.Finding(
                TrainingReportDiagnostics.Severity.WARNING,
                code,
                message,
                evidence));
    }

    private static void addDiagnosticRegression(
            List<TrainingReportDiagnostics.Finding> findings,
            TrainingReport baseline,
            TrainingReport candidate) {
        int baselineSeverity = severityRank(baseline.highestDiagnosticSeverity());
        int candidateSeverity = severityRank(candidate.highestDiagnosticSeverity());
        if (candidateSeverity <= baselineSeverity || candidateSeverity < 0) {
            return;
        }
        TrainingReportDiagnostics.Severity severity = candidateSeverity >= TrainingReportDiagnostics.Severity.CRITICAL.ordinal()
                ? TrainingReportDiagnostics.Severity.CRITICAL
                : candidateSeverity == TrainingReportDiagnostics.Severity.WARNING.ordinal()
                        ? TrainingReportDiagnostics.Severity.WARNING
                        : TrainingReportDiagnostics.Severity.INFO;
        findings.add(new TrainingReportDiagnostics.Finding(
                severity,
                "comparison.diagnostics.regressed",
                "Candidate report diagnostics are more severe than the baseline.",
                Map.of(
                        "baselineHighestSeverity", baseline.highestDiagnosticSeverity(),
                        "candidateHighestSeverity", candidate.highestDiagnosticSeverity(),
                        "candidateDiagnosticCodes", candidate.diagnosticSummary().codes())));
    }

    private static Optional<MetricDelta> metric(List<MetricDelta> metrics, String name) {
        return metrics.stream()
                .filter(metric -> metric.name().equals(name))
                .findFirst();
    }

    private static MetricDelta metric(
            String name,
            Direction direction,
            OptionalDouble baselineValue,
            OptionalDouble candidateValue) {
        return new MetricDelta(name, direction, baselineValue, candidateValue);
    }

    private static OptionalDouble positiveDuration(long durationMs) {
        return durationMs > 0L ? OptionalDouble.of(durationMs) : OptionalDouble.empty();
    }

    private static Map<String, Object> evidence(MetricDelta metric, double tolerance) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("metric", metric.name());
        evidence.put("direction", metric.direction().name());
        evidence.put("tolerance", tolerance);
        putOptional(evidence, "baseline", metric.baselineValue());
        putOptional(evidence, "candidate", metric.candidateValue());
        putOptional(evidence, "absoluteDelta", metric.absoluteDelta());
        putOptional(evidence, "relativeDelta", metric.relativeDelta());
        return Map.copyOf(evidence);
    }

    private static Map<String, Object> reportSummary(TrainingReport report) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("schema", report.schema());
        report.generatedAt().ifPresent(instant -> summary.put("generatedAt", instant.toString()));
        summary.put("epochCount", report.epochCount());
        summary.put("durationMs", report.durationMs());
        putOptional(summary, "bestValidationLoss", report.bestValidationLoss());
        putOptional(summary, "latestValidationLoss", report.latestValidationLoss());
        putOptional(summary, "latestTrainLoss", report.latestTrainLoss());
        putOptional(summary, "latestGradientL2Norm", report.latestGradientL2Norm());
        putOptional(summary, "latestParameterUpdateToParameterL2Ratio",
                report.latestParameterUpdateToParameterL2Ratio());
        summary.put("highestDiagnosticSeverity", report.highestDiagnosticSeverity());
        return Map.copyOf(summary);
    }

    private static double toleranceFor(String metricName, Options options) {
        if ("latestGeneralizationGap".equals(metricName)) {
            return options.generalizationGapRegressionTolerance();
        }
        if ("durationMs".equals(metricName)) {
            return 0.0;
        }
        return options.lossRegressionTolerance();
    }

    private static boolean isMetricRegressed(MetricDelta metric, Options options) {
        if ("durationMs".equals(metric.name())) {
            if (!metric.available() || metric.baselineValue().orElseThrow() <= 0.0) {
                return false;
            }
            return metric.candidateValue().orElseThrow() / metric.baselineValue().orElseThrow()
                    > options.durationRegressionRatio();
        }
        return metric.regressed(toleranceFor(metric.name(), options));
    }

    private static int severityRank(String severity) {
        if (severity == null || severity.isBlank() || "NONE".equalsIgnoreCase(severity)) {
            return -1;
        }
        for (TrainingReportDiagnostics.Severity value : TrainingReportDiagnostics.Severity.values()) {
            if (value.name().equalsIgnoreCase(severity)) {
                return value.ordinal();
            }
        }
        return -1;
    }

    private static void putOptional(Map<String, Object> target, String key, OptionalDouble value) {
        if (value != null && value.isPresent()) {
            target.put(key, value.getAsDouble());
        }
    }

    private static double nonNegativeOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value >= 0.0 ? value : fallback;
    }

    private static double ratioOrDefault(double value, double fallback) {
        return Double.isFinite(value) && value > 1.0 ? value : fallback;
    }
}
