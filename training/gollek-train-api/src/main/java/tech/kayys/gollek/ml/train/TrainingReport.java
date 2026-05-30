package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.longValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Type-safe view over a canonical trainer report payload.
 */
public record TrainingReport(Map<String, Object> payload) {
    public TrainingReport {
        Objects.requireNonNull(payload, "payload must not be null");
        payload = immutableMap(payload);
    }

    public static TrainingReport of(Map<String, ?> payload) {
        Objects.requireNonNull(payload, "payload must not be null");
        return new TrainingReport(immutableMap(payload));
    }

    public boolean canonical() {
        return TrainingReportReader.isCanonical(payload);
    }

    public String schema() {
        Object value = payload.get("schema");
        return value == null ? "" : String.valueOf(value);
    }

    public Optional<Instant> generatedAt() {
        Object value = payload.get("generatedAt");
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(Instant.parse(String.valueOf(value)));
        } catch (DateTimeParseException ignored) {
            return Optional.empty();
        }
    }

    public int epochCount() {
        return intValue(payload.get("epochCount"), 0);
    }

    public OptionalDouble bestValidationLoss() {
        return optionalDouble(payload.get("bestValidationLoss"));
    }

    public OptionalInt bestValidationEpoch() {
        int epoch = intValue(payload.get("bestValidationEpoch"), -1);
        return epoch >= 0 ? OptionalInt.of(epoch) : OptionalInt.empty();
    }

    public OptionalDouble latestTrainLoss() {
        return TrainingReportReader.latestTrainLoss(payload);
    }

    public OptionalDouble latestValidationLoss() {
        return TrainingReportReader.latestValidationLoss(payload);
    }

    public long durationMs() {
        return longValue(payload.get("durationMs"), 0L);
    }

    public List<Map<String, Object>> history() {
        return TrainingReportReader.history(payload);
    }

    public List<TrainingReportEpochSnapshot> epochSnapshots() {
        return TrainingReportReader.epochSnapshots(payload);
    }

    public Optional<TrainingReportEpochSnapshot> latestEpochSnapshot() {
        return TrainingReportReader.latestEpochSnapshot(payload);
    }

    public Optional<TrainingReportEpochSnapshot> epochSnapshot(int epoch) {
        return TrainingReportReader.epochSnapshot(payload, epoch);
    }

    public TrainingReportSeries trainLossSeries() {
        return TrainingReportReader.trainLossSeries(payload);
    }

    public TrainingReportSeries validationLossSeries() {
        return TrainingReportReader.validationLossSeries(payload);
    }

    public TrainingReportSeries learningRateSeries() {
        return TrainingReportReader.learningRateSeries(payload);
    }

    public TrainingReportSeries generalizationGapSeries() {
        return TrainingReportReader.generalizationGapSeries(payload);
    }

    public TrainingReportSeries validationToTrainLossRatioSeries() {
        return TrainingReportReader.validationToTrainLossRatioSeries(payload);
    }

    public TrainingReportSeries trainMetricSeries(String metricName) {
        return TrainingReportReader.trainMetricSeries(payload, metricName);
    }

    public TrainingReportSeries validationMetricSeries(String metricName) {
        return TrainingReportReader.validationMetricSeries(payload, metricName);
    }

    public TrainingReportSeries gradientL2NormSeries() {
        return TrainingReportReader.gradientL2NormSeries(payload);
    }

    public TrainingReportSeries parameterUpdateToParameterL2RatioSeries() {
        return TrainingReportReader.parameterUpdateToParameterL2RatioSeries(payload);
    }

    public TrainingReportSeriesBundle seriesBundle() {
        return TrainingReportReader.seriesBundle(payload);
    }

    public TrainingReportSeriesExport seriesExport() {
        return TrainingReportReader.seriesExport(payload);
    }

    public Map<String, Object> historySummary() {
        return TrainingReportReader.historySummary(payload);
    }

    public TrainingReportHistoryOverview historyOverview() {
        return TrainingReportReader.historyOverview(payload);
    }

    public Map<String, Object> trainLossSummary() {
        return TrainingReportReader.trainLossSummary(payload);
    }

    public TrainingReportLossSummary trainLoss() {
        return TrainingReportReader.trainLoss(payload);
    }

    public Map<String, Object> trainMetricSummaries() {
        return TrainingReportReader.trainMetricSummaries(payload);
    }

    public Map<String, TrainingReportMetricSummary> trainMetrics() {
        return TrainingReportReader.trainMetrics(payload);
    }

    public TrainingReportMetricSummary trainMetric(String metricName) {
        return TrainingReportReader.trainMetric(payload, metricName);
    }

    public Map<String, Object> validationLossSummary() {
        return TrainingReportReader.validationLossSummary(payload);
    }

    public TrainingReportLossSummary validationLoss() {
        return TrainingReportReader.validationLoss(payload);
    }

    public Map<String, Object> validationMetricSummaries() {
        return TrainingReportReader.validationMetricSummaries(payload);
    }

    public Map<String, TrainingReportMetricSummary> validationMetrics() {
        return TrainingReportReader.validationMetrics(payload);
    }

    public TrainingReportMetricSummary validationMetric(String metricName) {
        return TrainingReportReader.validationMetric(payload, metricName);
    }

    public Map<String, Object> generalizationSummary() {
        return TrainingReportReader.generalizationSummary(payload);
    }

    public TrainingReportGeneralizationSummary generalization() {
        return TrainingReportReader.generalization(payload);
    }

    public Map<String, Object> learningRateSummary() {
        return TrainingReportReader.learningRateSummary(payload);
    }

    public TrainingReportLearningRateSummary learningRate() {
        return TrainingReportReader.learningRate(payload);
    }

    public Map<String, Object> optimizationSummary() {
        return TrainingReportReader.optimizationSummary(payload);
    }

    public TrainingReportOptimizationSummary optimization() {
        return TrainingReportReader.optimization(payload);
    }

    public Map<String, Object> runHealthMap() {
        return TrainingReportReader.runHealth(payload);
    }

    public TrainingReportRunHealth runHealth() {
        return TrainingReportReader.runHealthView(payload);
    }

    public Map<String, Object> dataHealthMap() {
        return TrainingReportReader.dataHealth(payload);
    }

    public TrainingReportDataHealth dataHealth() {
        return TrainingReportReader.dataHealthView(payload);
    }

    public List<Map<String, Object>> diagnostics() {
        return TrainingReportReader.diagnostics(payload);
    }

    public List<TrainingReportDiagnostics.Finding> diagnosticFindings() {
        return TrainingReportReader.diagnosticFindings(payload);
    }

    public Map<String, Object> diagnosticsSummary() {
        return TrainingReportReader.diagnosticsSummary(payload);
    }

    public TrainingReportDiagnostics.Summary diagnosticSummary() {
        return TrainingReportReader.diagnosticSummary(payload);
    }

    public Map<String, Object> diagnosticsProvenance() {
        return TrainingReportReader.diagnosticsProvenance(payload);
    }

    public TrainingReportDiagnostics.Provenance diagnosticProvenance() {
        return TrainingReportReader.diagnosticProvenance(payload);
    }

    public TrainingReportDiagnostics.GateResult diagnosticGate(
            TrainingReportDiagnostics.Severity maxAllowedSeverity) {
        return TrainingReportDiagnostics.gateFindings(diagnosticFindings(), maxAllowedSeverity);
    }

    public TrainingReportValidationPolicy.Result validate() {
        return validate(TrainingReportValidationPolicy.defaultPolicy());
    }

    public TrainingReportValidationPolicy.Result validate(TrainingReportValidationPolicy policy) {
        TrainingReportValidationPolicy resolvedPolicy = policy == null
                ? TrainingReportValidationPolicy.defaultPolicy()
                : policy;
        return resolvedPolicy.validate(this);
    }

    public List<TrainingReportRecommendation> recommendations() {
        return TrainingReportReader.recommendations(payload);
    }

    public TrainingReportActionPlan actionPlan() {
        return TrainingReportReader.actionPlan(payload);
    }

    public String actionPlanMarkdown() {
        return TrainingReportActionPlanMarkdown.render(this);
    }

    public TrainingReportComparison compareCandidate(TrainingReport candidate) {
        return TrainingReportComparison.compare(this, candidate);
    }

    public String highestDiagnosticSeverity() {
        return diagnosticSummary().highestSeverity();
    }

    public boolean hasDiagnosticWarnings() {
        return diagnosticSummary().hasWarnings();
    }

    public boolean hasCriticalDiagnostics() {
        return diagnosticSummary().hasCritical();
    }

    public OptionalDouble latestGeneralizationGap() {
        return TrainingReportReader.latestGeneralizationGap(payload);
    }

    public OptionalDouble latestValidationToTrainLossRatio() {
        return TrainingReportReader.latestValidationToTrainLossRatio(payload);
    }

    public OptionalDouble latestLearningRate() {
        return TrainingReportReader.latestLearningRate(payload);
    }

    public OptionalDouble latestTrainMetric(String metricName) {
        return TrainingReportReader.latestTrainMetric(payload, metricName);
    }

    public OptionalDouble latestValidationMetric(String metricName) {
        return TrainingReportReader.latestValidationMetric(payload, metricName);
    }

    public OptionalDouble latestGradientL2Norm() {
        return TrainingReportReader.latestGradientL2Norm(payload);
    }

    public OptionalDouble latestParameterUpdateToParameterL2Ratio() {
        return TrainingReportReader.latestParameterUpdateToParameterL2Ratio(payload);
    }

    public Map<String, Object> metadata() {
        return TrainingReportReader.metadata(payload);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
