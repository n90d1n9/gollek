package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Reads and exposes persisted canonical trainer reports.
 */
public final class TrainingReportReader {
    public static final String CANONICAL_SCHEMA = TrainerTrainingReport.SCHEMA;

    private TrainingReportReader() {
    }

    public static Map<String, Object> read(Path reportFile) throws IOException {
        Objects.requireNonNull(reportFile, "reportFile must not be null");
        String json = Files.readString(reportFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid training report JSON at " + reportFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object training report at " + reportFile);
        }
        return immutableMap(map);
    }

    public static Map<String, Object> readCanonical(Path reportFile) throws IOException {
        Map<String, Object> report = read(reportFile);
        if (!isCanonical(report)) {
            Object schema = report.get("schema");
            throw new IOException("Unsupported training report schema at " + reportFile + ": " + schema);
        }
        return report;
    }

    public static TrainingReport readReport(Path reportFile) throws IOException {
        return TrainingReport.of(readCanonical(reportFile));
    }

    public static boolean isCanonical(Map<String, ?> report) {
        return report != null && CANONICAL_SCHEMA.equals(report.get("schema"));
    }

    public static List<Map<String, Object>> history(Map<String, ?> report) {
        Object value = requireReport(report).get("history");
        if (value == null) {
            return List.of();
        }
        if (!(value instanceof Iterable<?> iterable)) {
            throw new IllegalArgumentException("training report history must be an array");
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : iterable) {
            if (!(item instanceof Map<?, ?> row)) {
                throw new IllegalArgumentException("training report history rows must be objects");
            }
            rows.add(immutableMap(row));
        }
        return List.copyOf(rows);
    }

    public static List<TrainingReportEpochSnapshot> epochSnapshots(Map<String, ?> report) {
        return TrainingReportEpochSnapshot.fromMaps(history(report));
    }

    public static Optional<TrainingReportEpochSnapshot> latestEpochSnapshot(Map<String, ?> report) {
        List<TrainingReportEpochSnapshot> snapshots = epochSnapshots(report);
        if (snapshots.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(snapshots.get(snapshots.size() - 1));
    }

    public static Optional<TrainingReportEpochSnapshot> epochSnapshot(Map<String, ?> report, int epoch) {
        return epochSnapshots(report).stream()
                .filter(snapshot -> snapshot.epoch().isPresent() && snapshot.epoch().getAsInt() == epoch)
                .findFirst();
    }

    public static TrainingReportSeries trainLossSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots("trainLoss", epochSnapshots(report), TrainingReportEpochSnapshot::trainLoss);
    }

    public static TrainingReportSeries validationLossSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "validationLoss",
                epochSnapshots(report),
                TrainingReportEpochSnapshot::validationLoss);
    }

    public static TrainingReportSeries learningRateSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "learningRate",
                epochSnapshots(report),
                TrainingReportEpochSnapshot::learningRate);
    }

    public static TrainingReportSeries generalizationGapSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "generalizationGap",
                epochSnapshots(report),
                TrainingReportEpochSnapshot::generalizationGap);
    }

    public static TrainingReportSeries validationToTrainLossRatioSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "validationToTrainLossRatio",
                epochSnapshots(report),
                TrainingReportEpochSnapshot::validationToTrainLossRatio);
    }

    public static TrainingReportSeries trainMetricSeries(Map<String, ?> report, String metricName) {
        Objects.requireNonNull(metricName, "metricName must not be null");
        return TrainingReportSeries.fromSnapshots(
                "trainMetric." + metricName,
                epochSnapshots(report),
                snapshot -> snapshot.trainMetric(metricName));
    }

    public static TrainingReportSeries validationMetricSeries(Map<String, ?> report, String metricName) {
        Objects.requireNonNull(metricName, "metricName must not be null");
        return TrainingReportSeries.fromSnapshots(
                "validationMetric." + metricName,
                epochSnapshots(report),
                snapshot -> snapshot.validationMetric(metricName));
    }

    public static TrainingReportSeries gradientL2NormSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "gradientL2Norm",
                epochSnapshots(report),
                snapshot -> snapshot.optimization().gradientL2Norm());
    }

    public static TrainingReportSeries parameterUpdateToParameterL2RatioSeries(Map<String, ?> report) {
        return TrainingReportSeries.fromSnapshots(
                "parameterUpdateToParameterL2Ratio",
                epochSnapshots(report),
                snapshot -> snapshot.optimization().parameterUpdateToParameterL2Ratio());
    }

    public static TrainingReportSeriesBundle seriesBundle(Map<String, ?> report) {
        return TrainingReportSeriesBundle.fromSnapshots(epochSnapshots(report));
    }

    public static TrainingReportSeriesExport seriesExport(Map<String, ?> report) {
        return TrainingReportSeriesExport.fromBundle(seriesBundle(report));
    }

    public static Map<String, Object> historySummary(Map<String, ?> report) {
        return mapSection(report, "historySummary");
    }

    public static TrainingReportHistoryOverview historyOverview(Map<String, ?> report) {
        return TrainingReportHistoryOverview.fromMap(historySummary(report));
    }

    public static Map<String, Object> trainLossSummary(Map<String, ?> report) {
        return historySummarySection(report, "trainLoss");
    }

    public static TrainingReportLossSummary trainLoss(Map<String, ?> report) {
        return TrainingReportLossSummary.fromMap(trainLossSummary(report));
    }

    public static Map<String, Object> trainMetricSummaries(Map<String, ?> report) {
        return historySummarySection(report, "trainMetrics");
    }

    public static Map<String, TrainingReportMetricSummary> trainMetrics(Map<String, ?> report) {
        return TrainingReportMetricSummary.fromMaps(trainMetricSummaries(report));
    }

    public static TrainingReportMetricSummary trainMetric(Map<String, ?> report, String metricName) {
        Objects.requireNonNull(metricName, "metricName must not be null");
        return TrainingReportMetricSummary.fromMap(metricName, metricSummary(trainMetricSummaries(report), metricName));
    }

    public static Map<String, Object> validationLossSummary(Map<String, ?> report) {
        return historySummarySection(report, "validationLoss");
    }

    public static TrainingReportLossSummary validationLoss(Map<String, ?> report) {
        return TrainingReportLossSummary.fromMap(validationLossSummary(report));
    }

    public static Map<String, Object> validationMetricSummaries(Map<String, ?> report) {
        return historySummarySection(report, "validationMetrics");
    }

    public static Map<String, TrainingReportMetricSummary> validationMetrics(Map<String, ?> report) {
        return TrainingReportMetricSummary.fromMaps(validationMetricSummaries(report));
    }

    public static TrainingReportMetricSummary validationMetric(Map<String, ?> report, String metricName) {
        Objects.requireNonNull(metricName, "metricName must not be null");
        return TrainingReportMetricSummary.fromMap(
                metricName,
                metricSummary(validationMetricSummaries(report), metricName));
    }

    public static Map<String, Object> generalizationSummary(Map<String, ?> report) {
        return historySummarySection(report, "generalization");
    }

    public static TrainingReportGeneralizationSummary generalization(Map<String, ?> report) {
        return TrainingReportGeneralizationSummary.fromMap(generalizationSummary(report));
    }

    public static Map<String, Object> learningRateSummary(Map<String, ?> report) {
        return historySummarySection(report, "learningRate");
    }

    public static TrainingReportLearningRateSummary learningRate(Map<String, ?> report) {
        return TrainingReportLearningRateSummary.fromMap(learningRateSummary(report));
    }

    public static Map<String, Object> optimizationSummary(Map<String, ?> report) {
        Object value = historySummary(report).get("optimization");
        if (value instanceof Map<?, ?> map) {
            return immutableMap(map);
        }
        return Map.of();
    }

    public static TrainingReportOptimizationSummary optimization(Map<String, ?> report) {
        return TrainingReportOptimizationSummary.fromMap(optimizationSummary(report));
    }

    public static Map<String, Object> metadata(Map<String, ?> report) {
        return mapSection(report, "metadata");
    }

    public static Map<String, Object> runHealth(Map<String, ?> report) {
        Object value = requireReport(report).get("runHealth");
        if (value instanceof Map<?, ?> map) {
            return immutableMap(map);
        }
        return TrainingReportRunHealth.fromMetadata(metadata(report)).toMap();
    }

    public static TrainingReportRunHealth runHealthView(Map<String, ?> report) {
        return TrainingReportRunHealth.fromMap(runHealth(report));
    }

    public static Map<String, Object> dataHealth(Map<String, ?> report) {
        Object value = requireReport(report).get("dataHealth");
        if (value instanceof Map<?, ?> map) {
            return immutableMap(map);
        }
        return TrainingReportDataHealth.fromMetadata(metadata(report)).toMap();
    }

    public static TrainingReportDataHealth dataHealthView(Map<String, ?> report) {
        return TrainingReportDataHealth.fromMap(dataHealth(report));
    }

    public static List<Map<String, Object>> diagnostics(Map<String, ?> report) {
        return TrainingReportDiagnostics.toMaps(diagnosticFindings(report));
    }

    public static List<TrainingReportDiagnostics.Finding> diagnosticFindings(Map<String, ?> report) {
        return TrainingReportDiagnostics.withReportHealthFindings(
                report,
                TrainingReportDiagnostics.fromMaps(rawDiagnostics(report)));
    }

    public static Map<String, Object> diagnosticsSummary(Map<String, ?> report) {
        return diagnosticSummary(report).toMap();
    }

    public static TrainingReportDiagnostics.Summary diagnosticSummary(Map<String, ?> report) {
        return TrainingReportDiagnostics.summarize(diagnosticFindings(report));
    }

    public static Map<String, Object> diagnosticsProvenance(Map<String, ?> report) {
        return diagnosticProvenance(report).toMap();
    }

    public static TrainingReportDiagnostics.Provenance diagnosticProvenance(Map<String, ?> report) {
        Objects.requireNonNull(report, "report must not be null");
        List<TrainingReportDiagnostics.Finding> persistedFindings =
                TrainingReportDiagnostics.fromMaps(rawDiagnostics(report));
        List<TrainingReportDiagnostics.Finding> effectiveFindings =
                TrainingReportDiagnostics.withReportHealthFindings(report, persistedFindings);
        TrainingReportDiagnostics.Summary persistedFromFindings =
                TrainingReportDiagnostics.summarize(persistedFindings);
        TrainingReportDiagnostics.Summary effectiveSummary =
                TrainingReportDiagnostics.summarize(effectiveFindings);

        Object rawSummary = requireReport(report).get("diagnosticsSummary");
        boolean summaryPersisted = rawSummary instanceof Map<?, ?>;
        TrainingReportDiagnostics.Summary persistedSummary = summaryPersisted
                ? TrainingReportDiagnostics.Summary.fromMap(immutableMap((Map<?, ?>) rawSummary))
                : persistedFromFindings;
        return new TrainingReportDiagnostics.Provenance(
                requireReport(report).get("diagnostics") instanceof Iterable<?>,
                summaryPersisted,
                hasBackfilledFindings(persistedFindings, effectiveFindings),
                summaryPersisted && !sameSummary(persistedSummary, effectiveSummary),
                persistedFindings.size(),
                effectiveFindings.size(),
                persistedSummary.highestSeverity(),
                effectiveSummary.highestSeverity(),
                diagnosticCodes(persistedFindings),
                diagnosticCodes(effectiveFindings),
                backfilledCodes(persistedFindings, effectiveFindings));
    }

    public static TrainingReportValidationPolicy.Result validate(Map<String, ?> report) {
        return validate(report, TrainingReportValidationPolicy.defaultPolicy());
    }

    public static TrainingReportValidationPolicy.Result validate(
            Map<String, ?> report,
            TrainingReportValidationPolicy policy) {
        TrainingReportValidationPolicy resolvedPolicy = policy == null
                ? TrainingReportValidationPolicy.defaultPolicy()
                : policy;
        return TrainingReport.of(requireReport(report)).validate(resolvedPolicy);
    }

    public static List<TrainingReportRecommendation> recommendations(Map<String, ?> report) {
        return TrainingReportAdvisor.recommendations(report);
    }

    public static TrainingReportActionPlan actionPlan(Map<String, ?> report) {
        return TrainingReportAdvisor.actionPlan(report);
    }

    public static OptionalDouble latestTrainLoss(Map<String, ?> report) {
        return optionalDouble(requireReport(report).get("latestTrainLoss"));
    }

    public static OptionalDouble latestValidationLoss(Map<String, ?> report) {
        return optionalDouble(requireReport(report).get("latestValidationLoss"));
    }

    public static OptionalDouble latestGeneralizationGap(Map<String, ?> report) {
        return optionalDouble(generalizationSummary(report).get("latestGap"));
    }

    public static OptionalDouble latestValidationToTrainLossRatio(Map<String, ?> report) {
        return optionalDouble(generalizationSummary(report).get("latestValidationToTrainLossRatio"));
    }

    public static OptionalDouble latestLearningRate(Map<String, ?> report) {
        return optionalDouble(learningRateSummary(report).get("latest"));
    }

    public static OptionalDouble latestTrainMetric(Map<String, ?> report, String metricName) {
        return trainMetric(report, metricName).latest();
    }

    public static OptionalDouble latestValidationMetric(Map<String, ?> report, String metricName) {
        return validationMetric(report, metricName).latest();
    }

    public static OptionalDouble latestGradientL2Norm(Map<String, ?> report) {
        return nestedOptionalDouble(optimizationSummary(report), "latest", "gradientL2Norm");
    }

    public static OptionalDouble latestParameterUpdateToParameterL2Ratio(Map<String, ?> report) {
        return nestedOptionalDouble(optimizationSummary(report), "latest", "parameterUpdateToParameterL2Ratio");
    }

    private static Map<String, ?> requireReport(Map<String, ?> report) {
        return Objects.requireNonNull(report, "report must not be null");
    }

    private static List<Map<String, Object>> rawDiagnostics(Map<String, ?> report) {
        return mapListSection(report, "diagnostics");
    }

    private static boolean hasBackfilledFindings(
            List<TrainingReportDiagnostics.Finding> persisted,
            List<TrainingReportDiagnostics.Finding> effective) {
        return !backfilledCodes(persisted, effective).isEmpty();
    }

    private static boolean sameSummary(
            TrainingReportDiagnostics.Summary left,
            TrainingReportDiagnostics.Summary right) {
        return left.total() == right.total()
                && left.highestSeverity().equals(right.highestSeverity())
                && left.codes().equals(right.codes());
    }

    private static List<String> diagnosticCodes(List<TrainingReportDiagnostics.Finding> findings) {
        if (findings == null || findings.isEmpty()) {
            return List.of();
        }
        return findings.stream()
                .map(TrainingReportDiagnostics.Finding::code)
                .toList();
    }

    private static List<String> backfilledCodes(
            List<TrainingReportDiagnostics.Finding> persisted,
            List<TrainingReportDiagnostics.Finding> effective) {
        List<String> remainingPersisted = new ArrayList<>(diagnosticCodes(persisted));
        List<String> backfilled = new ArrayList<>();
        for (String code : diagnosticCodes(effective)) {
            int index = remainingPersisted.indexOf(code);
            if (index >= 0) {
                remainingPersisted.remove(index);
            } else if (!backfilled.contains(code)) {
                backfilled.add(code);
            }
        }
        return List.copyOf(backfilled);
    }

    private static Map<String, Object> historySummarySection(Map<String, ?> report, String key) {
        Object value = historySummary(report).get(key);
        if (value instanceof Map<?, ?> map) {
            return immutableMap(map);
        }
        return Map.of();
    }

    private static Map<String, Object> metricSummary(Map<String, Object> summaries, String metricName) {
        Object value = summaries.get(metricName);
        if (value instanceof Map<?, ?> map) {
            return immutableMap(map);
        }
        return Map.of();
    }

    private static Map<String, Object> mapSection(Map<String, ?> report, String key) {
        Object value = requireReport(report).get(key);
        if (value == null) {
            return Map.of();
        }
        if (!(value instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("training report " + key + " must be an object");
        }
        return immutableMap(map);
    }

    private static List<Map<String, Object>> mapListSection(Map<String, ?> report, String key) {
        Object value = requireReport(report).get(key);
        if (value == null) {
            return List.of();
        }
        if (!(value instanceof Iterable<?> iterable)) {
            throw new IllegalArgumentException("training report " + key + " must be an array");
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : iterable) {
            if (!(item instanceof Map<?, ?> row)) {
                throw new IllegalArgumentException("training report " + key + " rows must be objects");
            }
            rows.add(immutableMap(row));
        }
        return List.copyOf(rows);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableMap(Map<?, ?> map) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static OptionalDouble nestedOptionalDouble(
            Map<String, Object> section,
            String nestedKey,
            String valueKey) {
        Object nested = section.get(nestedKey);
        if (!(nested instanceof Map<?, ?> nestedMap)) {
            return OptionalDouble.empty();
        }
        return optionalDouble(nestedMap.get(valueKey));
    }
}
