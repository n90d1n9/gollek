package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportReaderTest {
    @TempDir
    Path tempDir;

    @Test
    @SuppressWarnings("unchecked")
    void readsCanonicalReportAndExposesActionableSections() throws IOException {
        Path reportFile = tempDir.resolve("canonical-report.json");
        Map<String, Object> metadata = Map.of(
                "device", "metal",
                "epochHistory", List.of(
                        Map.ofEntries(
                                Map.entry("epoch", 0),
                                Map.entry("trainLoss", 0.8),
                                Map.entry("validationLoss", 1.0),
                                Map.entry("learningRate", 0.01),
                                Map.entry("gradientL2Norm", 0.7),
                                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                                Map.entry("parameterUpdateToParameterL2Ratio", 0.01),
                                Map.entry("trainMetrics", Map.of("accuracy", 0.45)),
                                Map.entry("validationMetrics", Map.of("accuracy", 0.5))),
                        Map.ofEntries(
                                Map.entry("epoch", 1),
                                Map.entry("trainLoss", 0.5),
                                Map.entry("validationLoss", 0.65),
                                Map.entry("learningRate", 0.001),
                                Map.entry("gradientL2Norm", 0.3),
                                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                                Map.entry("parameterUpdateToParameterL2Ratio", 0.03),
                                Map.entry("trainMetrics", Map.of("accuracy", 0.7)),
                                Map.entry("validationMetrics", Map.of("accuracy", 0.75)))));
        TrainingSummary summary = new TrainingSummary(
                2,
                0.65,
                1,
                0.5,
                0.65,
                123L,
                metadata);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T10:11:12Z")))
                        + "\n",
                StandardCharsets.UTF_8);

        Map<String, Object> report = TrainingReportReader.readCanonical(reportFile);

        assertEquals(TrainingReportReader.CANONICAL_SCHEMA, report.get("schema"));
        assertTrue(TrainingReportReader.isCanonical(report));
        assertEquals(2, TrainingReportReader.history(report).size());
        List<TrainingReportEpochSnapshot> snapshots = TrainingReportReader.epochSnapshots(report);
        assertEquals(2, snapshots.size());
        TrainingReportEpochSnapshot latestSnapshot = TrainingReportReader.latestEpochSnapshot(report).orElseThrow();
        assertEquals(1, latestSnapshot.epoch().orElseThrow());
        assertEquals(0.5, latestSnapshot.trainLoss().orElseThrow(), 1e-12);
        assertEquals(0.65, latestSnapshot.validationLoss().orElseThrow(), 1e-12);
        assertEquals(0.15, latestSnapshot.generalizationGap().orElseThrow(), 1e-12);
        assertEquals(1.3, latestSnapshot.validationToTrainLossRatio().orElseThrow(), 1e-12);
        assertEquals(0.001, latestSnapshot.learningRate().orElseThrow(), 1e-12);
        assertEquals(0.7, latestSnapshot.trainMetric("accuracy").orElseThrow(), 1e-12);
        assertEquals(0.75, latestSnapshot.validationMetric("accuracy").orElseThrow(), 1e-12);
        assertTrue(latestSnapshot.hasTraining());
        assertTrue(latestSnapshot.hasValidation());
        assertEquals(0.3, latestSnapshot.optimization().gradientL2Norm().orElseThrow(), 1e-12);
        assertTrue(latestSnapshot.optimization().hasGradientDiagnostics());
        assertTrue(latestSnapshot.optimization().hasParameterUpdateDiagnostics());
        assertEquals(0.03, latestSnapshot.optimization()
                .parameterUpdateToParameterL2Ratio()
                .orElseThrow(), 1e-12);
        assertEquals(false, latestSnapshot.mixedPrecision().enabled());
        assertEquals(latestSnapshot, TrainingReportReader.epochSnapshot(report, 1).orElseThrow());
        assertTrue(TrainingReportReader.epochSnapshot(report, 99).isEmpty());
        assertThrows(UnsupportedOperationException.class,
                () -> latestSnapshot.trainMetrics().put("loss", 1.0));
        assertThrows(UnsupportedOperationException.class,
                () -> latestSnapshot.raw().put("extra", true));

        TrainingReportSeries trainLossSeries = TrainingReportReader.trainLossSeries(report);
        assertEquals("trainLoss", trainLossSeries.name());
        assertTrue(trainLossSeries.available());
        assertEquals(2, trainLossSeries.count());
        assertEquals(0.8, trainLossSeries.first().orElseThrow(), 1e-12);
        assertEquals(0.5, trainLossSeries.latest().orElseThrow(), 1e-12);
        assertEquals(-0.3, trainLossSeries.deltaFromFirst().orElseThrow(), 1e-12);
        assertEquals("decreased", trainLossSeries.trend());
        assertTrue(trainLossSeries.decreased());
        assertEquals(List.of(0.8, 0.5), trainLossSeries.values());
        assertEquals(0.5, trainLossSeries.toMap().get(1), 1e-12);
        assertThrows(UnsupportedOperationException.class,
                () -> trainLossSeries.points().add(new TrainingReportSeries.Point(2, 0.4)));
        assertThrows(UnsupportedOperationException.class,
                () -> trainLossSeries.toMap().put(2, 0.4));

        TrainingReportSeries validationLossSeries = TrainingReportReader.validationLossSeries(report);
        assertEquals(0.65, validationLossSeries.latest().orElseThrow(), 1e-12);
        assertEquals(0.65, validationLossSeries.min().orElseThrow(), 1e-12);
        assertEquals(1, validationLossSeries.minEpoch().orElseThrow());

        TrainingReportSeries learningRateSeries = TrainingReportReader.learningRateSeries(report);
        assertEquals(0.001, learningRateSeries.latest().orElseThrow(), 1e-12);
        assertTrue(learningRateSeries.decreased());

        TrainingReportSeries gapSeries = TrainingReportReader.generalizationGapSeries(report);
        assertEquals(0.15, gapSeries.latest().orElseThrow(), 1e-12);
        assertTrue(gapSeries.decreased());

        TrainingReportSeries trainAccuracySeries = TrainingReportReader.trainMetricSeries(report, "accuracy");
        assertEquals("trainMetric.accuracy", trainAccuracySeries.name());
        assertEquals(0.7, trainAccuracySeries.latest().orElseThrow(), 1e-12);
        assertTrue(trainAccuracySeries.increased());
        assertEquals(0.75, TrainingReportReader.validationMetricSeries(report, "accuracy")
                .latest()
                .orElseThrow(), 1e-12);
        assertEquals(0.3, TrainingReportReader.gradientL2NormSeries(report).latest().orElseThrow(), 1e-12);
        assertEquals(0.03, TrainingReportReader.parameterUpdateToParameterL2RatioSeries(report)
                .latest()
                .orElseThrow(), 1e-12);

        TrainingReportSeriesBundle seriesBundle = TrainingReportReader.seriesBundle(report);
        assertEquals(0.5, seriesBundle.trainLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.65, seriesBundle.validationLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.001, seriesBundle.learningRate().latest().orElseThrow(), 1e-12);
        assertEquals(0.15, seriesBundle.generalizationGap().latest().orElseThrow(), 1e-12);
        assertEquals(1.3, seriesBundle.validationToTrainLossRatio().latest().orElseThrow(), 1e-12);
        assertEquals(0.3, seriesBundle.gradientL2Norm().latest().orElseThrow(), 1e-12);
        assertEquals(0.03, seriesBundle.parameterUpdateToParameterL2Ratio().latest().orElseThrow(), 1e-12);
        assertTrue(seriesBundle.hasTrainMetrics());
        assertTrue(seriesBundle.hasValidationMetrics());
        assertTrue(seriesBundle.hasOptimization());
        assertEquals(0.7, seriesBundle.trainMetric("accuracy").latest().orElseThrow(), 1e-12);
        assertEquals(0.75, seriesBundle.validationMetric("accuracy").latest().orElseThrow(), 1e-12);
        assertFalse(seriesBundle.trainMetric("missing").available());
        assertTrue(seriesBundle.availableSeries().containsKey("trainLoss"));
        assertTrue(seriesBundle.availableSeries().containsKey("validationMetric.accuracy"));
        assertTrue(seriesBundle.allSeries().containsKey("learningRate"));
        assertThrows(UnsupportedOperationException.class,
                () -> seriesBundle.trainMetrics().put("loss", trainLossSeries));
        assertThrows(UnsupportedOperationException.class,
                () -> seriesBundle.availableSeries().put("extra", trainLossSeries));

        TrainingReportSeriesExport seriesExport = TrainingReportReader.seriesExport(report);
        assertTrue(seriesExport.available());
        assertEquals(2, seriesExport.epochCount());
        assertEquals(seriesExport.points().size(), seriesExport.pointCount());
        assertTrue(seriesExport.seriesNames().contains("trainLoss"));
        assertTrue(seriesExport.seriesNames().contains("validationMetric.accuracy"));
        assertEquals(TrainingReportSeriesExport.SCHEMA, seriesExport.toMap().get("schema"));
        assertTrue(seriesExport.toJson().contains("\"seriesNames\""));
        assertTrue(seriesExport.toCsv().startsWith("epoch,"));
        assertTrue(seriesExport.toCsv().contains("trainLoss"));
        assertTrue(seriesExport.toCsv().contains("validationMetric.accuracy"));
        assertTrue(seriesExport.toLongCsv().startsWith("series,epoch,value"));
        assertTrue(seriesExport.toLongCsv().contains("trainLoss,1,0.5"));
        assertThrows(UnsupportedOperationException.class,
                () -> seriesExport.rows().get(0).put("extra", 1.0));
        assertThrows(UnsupportedOperationException.class,
                () -> seriesExport.points().add(Map.of("series", "extra")));
        assertEquals(0.5, TrainingReportReader.latestTrainLoss(report).orElseThrow(), 1e-12);
        assertEquals(0.65, TrainingReportReader.latestValidationLoss(report).orElseThrow(), 1e-12);
        assertEquals(0.15, TrainingReportReader.latestGeneralizationGap(report).orElseThrow(), 1e-12);
        assertEquals(1.3, TrainingReportReader.latestValidationToTrainLossRatio(report).orElseThrow(), 1e-12);
        assertEquals(0.001, TrainingReportReader.latestLearningRate(report).orElseThrow(), 1e-12);

        Map<String, Object> historySummary = TrainingReportReader.historySummary(report);
        Map<String, Object> validationMetrics = (Map<String, Object>) historySummary.get("validationMetrics");
        Map<String, Object> accuracy = (Map<String, Object>) validationMetrics.get("accuracy");
        assertEquals("increased", accuracy.get("trend"));

        TrainingReportHistoryOverview historyOverview = TrainingReportReader.historyOverview(report);
        assertTrue(historyOverview.available());
        assertEquals(2, historyOverview.size());
        assertEquals(0, historyOverview.firstEpoch().orElseThrow());
        assertEquals(1, historyOverview.lastEpoch().orElseThrow());
        assertEquals(0.5, historyOverview.trainLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.65, historyOverview.validationLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.7, historyOverview.trainMetrics().get("accuracy").latest().orElseThrow(), 1e-12);
        assertEquals(0.75, historyOverview.validationMetrics().get("accuracy").latest().orElseThrow(), 1e-12);
        assertTrue(historyOverview.hasTrainMetrics());
        assertTrue(historyOverview.hasValidationMetrics());
        assertEquals(0.001, historyOverview.learningRate().latest().orElseThrow(), 1e-12);
        assertEquals(0.15, historyOverview.generalization().latestGap().orElseThrow(), 1e-12);
        assertEquals(0.3, historyOverview.optimization().latest().gradientL2Norm().orElseThrow(), 1e-12);
        assertThrows(UnsupportedOperationException.class,
                () -> historyOverview.trainMetrics().put("loss", TrainingReportMetricSummary.empty("loss")));

        TrainingReportMetricSummary trainAccuracy = TrainingReportReader.trainMetric(report, "accuracy");
        assertTrue(trainAccuracy.available());
        assertTrue(trainAccuracy.hasName("accuracy"));
        assertEquals(2, trainAccuracy.count());
        assertEquals(0.45, trainAccuracy.first().orElseThrow(), 1e-12);
        assertEquals(0, trainAccuracy.firstEpoch().orElseThrow());
        assertEquals(0.7, trainAccuracy.latest().orElseThrow(), 1e-12);
        assertEquals(1, trainAccuracy.latestEpoch().orElseThrow());
        assertEquals(0.25, trainAccuracy.deltaFromFirst().orElseThrow(), 1e-12);
        assertEquals("increased", trainAccuracy.trend());
        assertTrue(trainAccuracy.increased());
        assertTrue(trainAccuracy.latestIsMax());
        assertEquals(0.7, TrainingReportReader.latestTrainMetric(report, "accuracy").orElseThrow(), 1e-12);
        assertEquals(trainAccuracy, TrainingReportReader.trainMetrics(report).get("accuracy"));
        assertThrows(UnsupportedOperationException.class,
                () -> TrainingReportReader.trainMetrics(report).put("loss", trainAccuracy));

        TrainingReportMetricSummary validationAccuracy = TrainingReportReader.validationMetric(report, "accuracy");
        assertTrue(validationAccuracy.available());
        assertEquals(0.5, validationAccuracy.first().orElseThrow(), 1e-12);
        assertEquals(0.75, validationAccuracy.latest().orElseThrow(), 1e-12);
        assertEquals(0.5, validationAccuracy.min().orElseThrow(), 1e-12);
        assertEquals(0.75, validationAccuracy.max().orElseThrow(), 1e-12);
        assertEquals("increased", validationAccuracy.trend());
        assertTrue(validationAccuracy.increased());
        assertTrue(validationAccuracy.latestIsMax());
        assertEquals(0.75, TrainingReportReader.latestValidationMetric(report, "accuracy").orElseThrow(), 1e-12);
        assertEquals(validationAccuracy, TrainingReportReader.validationMetrics(report).get("accuracy"));
        assertFalse(TrainingReportReader.validationMetric(report, "missing").available());
        assertTrue(TrainingReportReader.latestValidationMetric(report, "missing").isEmpty());

        TrainingReportLossSummary trainLoss = TrainingReportReader.trainLoss(report);
        assertTrue(trainLoss.available());
        assertEquals(2, trainLoss.count());
        assertEquals(0.8, trainLoss.first().orElseThrow(), 1e-12);
        assertEquals(0.5, trainLoss.latest().orElseThrow(), 1e-12);
        assertEquals(0.5, trainLoss.best().orElseThrow(), 1e-12);
        assertEquals(1, trainLoss.bestEpoch().orElseThrow());
        assertEquals(-0.3, trainLoss.deltaFromFirst().orElseThrow(), 1e-12);
        assertEquals("improved", trainLoss.trend());
        assertTrue(trainLoss.improved());

        TrainingReportLossSummary validationLoss = TrainingReportReader.validationLoss(report);
        assertTrue(validationLoss.available());
        assertEquals(1.0, validationLoss.first().orElseThrow(), 1e-12);
        assertEquals(0.65, validationLoss.latest().orElseThrow(), 1e-12);
        assertEquals("improved", validationLoss.trend());

        TrainingReportGeneralizationSummary generalization = TrainingReportReader.generalization(report);
        assertTrue(generalization.available());
        assertEquals(2, generalization.count());
        assertEquals(0.2, generalization.firstGap().orElseThrow(), 1e-12);
        assertEquals(0.15, generalization.latestGap().orElseThrow(), 1e-12);
        assertEquals(-0.05, generalization.gapDeltaFromFirst().orElseThrow(), 1e-12);
        assertEquals("decreasing", generalization.gapTrend());
        assertTrue(generalization.narrowing());
        assertEquals(1.3, generalization.latestValidationToTrainLossRatio().orElseThrow(), 1e-12);
        assertEquals(true, generalization.latestValidationLossAboveTrainLoss());

        Map<String, Object> optimization = TrainingReportReader.optimizationSummary(report);
        Map<String, Object> latestOptimization = (Map<String, Object>) optimization.get("latest");
        assertEquals(0.3, TrainingReportReader.latestGradientL2Norm(report).orElseThrow(), 1e-12);
        assertEquals(0.03, TrainingReportReader.latestParameterUpdateToParameterL2Ratio(report).orElseThrow(), 1e-12);
        assertEquals(0.3, (double) latestOptimization.get("gradientL2Norm"), 1e-12);

        TrainingReportLearningRateSummary learningRate = TrainingReportReader.learningRate(report);
        assertTrue(learningRate.available());
        assertEquals(2, learningRate.count());
        assertEquals(0.01, learningRate.first().orElseThrow(), 1e-12);
        assertEquals(0, learningRate.firstEpoch().orElseThrow());
        assertEquals(0.001, learningRate.latest().orElseThrow(), 1e-12);
        assertEquals(1, learningRate.latestEpoch().orElseThrow());
        assertEquals(0.01, learningRate.best().orElseThrow(), 1e-12);
        assertEquals("decreased", learningRate.trend());
        assertTrue(learningRate.decreased());

        TrainingReportOptimizationSummary typedOptimization = TrainingReportReader.optimization(report);
        assertTrue(typedOptimization.available());
        assertEquals(2, typedOptimization.count());
        assertEquals(0, typedOptimization.firstEpoch().orElseThrow());
        assertEquals(1, typedOptimization.lastEpoch().orElseThrow());
        assertEquals(0.7, typedOptimization.gradients().l2Norm().first().orElseThrow(), 1e-12);
        assertEquals(0.3, typedOptimization.gradients().l2Norm().latest().orElseThrow(), 1e-12);
        assertEquals("decreased", typedOptimization.gradients().l2Norm().trend());
        assertTrue(typedOptimization.parameterUpdates().enabled());
        assertEquals(0.03, typedOptimization.parameterUpdates().toParameterL2Ratio().latest().orElseThrow(), 1e-12);
        assertEquals(1, typedOptimization.latest().epoch().orElseThrow());
        assertEquals(0.3, typedOptimization.latest().gradientL2Norm().orElseThrow(), 1e-12);
        assertEquals(0.03, typedOptimization.latest().parameterUpdateToParameterL2Ratio().orElseThrow(), 1e-12);

        assertEquals(0, TrainingReportReader.diagnosticsSummary(report).get("total"));
        assertEquals("NONE", TrainingReportReader.diagnosticsSummary(report).get("highestSeverity"));
        assertEquals("metal", TrainingReportReader.metadata(report).get("device"));
    }

    @Test
    void actionPlanTurnsDiagnosticsIntoTypedRecommendations() throws IOException {
        Path reportFile = tempDir.resolve("optimizer-report.json");
        Map<String, Object> row = Map.ofEntries(
                Map.entry("epoch", 0),
                Map.entry("trainLoss", 0.9),
                Map.entry("validationLoss", 0.85),
                Map.entry("gradientL2Norm", 0.4),
                Map.entry("gradientValueCount", 8L),
                Map.entry("gradientZeroFraction", 0.0),
                Map.entry("gradientNonFiniteCount", 1L),
                Map.entry("gradientNanCount", 1L),
                Map.entry("gradientPositiveInfinityCount", 0L),
                Map.entry("gradientNegativeInfinityCount", 0L),
                Map.entry("gradientNonFiniteFraction", 0.125),
                Map.entry("parameterNonFiniteCount", 0L),
                Map.entry("parameterNanCount", 0L),
                Map.entry("parameterPositiveInfinityCount", 0L),
                Map.entry("parameterNegativeInfinityCount", 0L),
                Map.entry("parameterNonFiniteFraction", 0.0),
                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                Map.entry("parameterUpdateL2Norm", 0.01),
                Map.entry("parameterUpdateNonFiniteCount", 0L),
                Map.entry("parameterUpdateNanCount", 0L),
                Map.entry("parameterUpdatePositiveInfinityCount", 0L),
                Map.entry("parameterUpdateNegativeInfinityCount", 0L),
                Map.entry("parameterUpdateNonFiniteFraction", 0.0),
                Map.entry("parameterUpdateToParameterL2Ratio", 0.01));
        TrainingSummary summary = new TrainingSummary(
                1,
                0.85,
                0,
                0.9,
                0.85,
                42L,
                Map.of("epochHistory", List.of(row)));
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T14:15:16Z"))),
                StandardCharsets.UTF_8);

        TrainingReport report = TrainingReportReader.readReport(reportFile);
        TrainingReportActionPlan actionPlan = report.actionPlan();

        assertEquals(TrainingReportActionPlan.Status.BLOCKED, actionPlan.status());
        assertTrue(actionPlan.hasBlockers());
        assertEquals(1, actionPlan.blockers().size());
        assertEquals("optimization.non_finite_values", actionPlan.blockers().get(0).diagnosticCode());
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, actionPlan.blockers().get(0).category());
        assertEquals(TrainingReportRecommendation.Priority.BLOCKER, actionPlan.blockers().get(0).priority());
        assertTrue(actionPlan.actionItems().stream().anyMatch(action -> action.contains("gradient clipping")));
        assertEquals(actionPlan, TrainingReportReader.actionPlan(report.payload()));
        assertEquals(actionPlan, Gollek.DL.trainingReportActionPlan(reportFile));
        assertEquals(1, Gollek.DL.trainingReportRecommendations(reportFile).size());
        assertThrows(UnsupportedOperationException.class, () -> actionPlan.toMap().put("status", "READY"));
    }

    @Test
    void typedTrainingReportExposesCanonicalFields() throws IOException {
        Path reportFile = tempDir.resolve("canonical-report.json");
        Map<String, Object> metadata = Map.of(
                "device", "metal",
                "epochHistory", List.of(
                        Map.of(
                                "epoch", 0,
                                "trainLoss", 0.9,
                                "validationLoss", 1.1,
                                "validationMetrics", Map.of("accuracy", 0.4)),
                        Map.of(
                                "epoch", 1,
                                "trainLoss", 0.6,
                                "validationLoss", 0.75,
                                "validationMetrics", Map.of("accuracy", 0.8))));
        TrainingSummary summary = new TrainingSummary(
                2,
                0.75,
                1,
                0.6,
                0.75,
                321L,
                metadata);
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T12:13:14Z"))),
                StandardCharsets.UTF_8);

        TrainingReport report = TrainingReportReader.readReport(reportFile);

        assertTrue(report.canonical());
        assertEquals(TrainingReportReader.CANONICAL_SCHEMA, report.schema());
        assertEquals(2, report.epochCount());
        assertEquals(321L, report.durationMs());
        assertEquals(0.75, report.bestValidationLoss().orElseThrow(), 1e-12);
        assertEquals(1, report.bestValidationEpoch().orElseThrow());
        assertEquals(0.6, report.latestTrainLoss().orElseThrow(), 1e-12);
        assertEquals(0.75, report.latestValidationLoss().orElseThrow(), 1e-12);
        assertEquals(0.15, report.latestGeneralizationGap().orElseThrow(), 1e-12);
        assertEquals(1.25, report.latestValidationToTrainLossRatio().orElseThrow(), 1e-12);
        assertTrue(report.trainLossSummary().containsKey("available"));
        assertTrue(report.historyOverview().available());
        assertEquals(2, report.historyOverview().size());
        assertEquals(0.75, report.historyOverview().validationLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.8, report.historyOverview().validationMetrics().get("accuracy").latest().orElseThrow(), 1e-12);
        assertEquals(false, report.historyOverview().hasTrainMetrics());
        assertEquals(true, report.historyOverview().hasValidationMetrics());
        assertTrue(report.trainLoss().available());
        assertEquals(0.6, report.trainLoss().latest().orElseThrow(), 1e-12);
        assertTrue(report.trainLoss().improved());
        assertTrue(report.validationLossSummary().containsKey("available"));
        assertTrue(report.validationLoss().available());
        assertEquals(0.75, report.validationLoss().latest().orElseThrow(), 1e-12);
        assertTrue(report.validationLoss().improved());
        assertTrue(report.trainMetrics().isEmpty());
        assertTrue(report.validationMetricSummaries().containsKey("accuracy"));
        assertEquals(1, report.validationMetrics().size());
        assertTrue(report.validationMetric("accuracy").available());
        assertEquals(0.8, report.validationMetric("accuracy").latest().orElseThrow(), 1e-12);
        assertEquals(0.8, report.latestValidationMetric("accuracy").orElseThrow(), 1e-12);
        assertTrue(report.validationMetric("accuracy").increased());
        assertTrue(report.validationMetric("missing").latest().isEmpty());
        assertTrue(report.generalizationSummary().containsKey("available"));
        assertTrue(report.generalization().available());
        assertEquals(0.15, report.generalization().latestGap().orElseThrow(), 1e-12);
        assertEquals(1.25, report.generalization().latestValidationToTrainLossRatio().orElseThrow(), 1e-12);
        assertTrue(report.learningRateSummary().containsKey("available"));
        assertEquals(false, report.learningRate().available());
        assertEquals(0, report.learningRate().count());
        assertTrue(report.learningRate().latest().isEmpty());
        assertTrue(report.latestLearningRate().isEmpty());
        assertTrue(report.optimizationSummary().containsKey("available"));
        assertEquals(false, report.optimization().available());
        assertEquals(0, report.optimization().count());
        assertTrue(report.optimization().latest().gradientL2Norm().isEmpty());
        assertTrue(report.latestGradientL2Norm().isEmpty());
        assertTrue(report.latestParameterUpdateToParameterL2Ratio().isEmpty());
        assertEquals("2026-05-18T12:13:14Z", report.generatedAt().orElseThrow().toString());
        assertEquals("NONE", report.diagnosticSummary().highestSeverity());
        assertEquals("FRESH", report.diagnosticProvenance().status());
        assertTrue(report.diagnosticProvenance().fresh());
        assertEquals(false, report.diagnosticProvenance().diagnosticsBackfilled());
        assertEquals("FRESH", report.diagnosticsProvenance().get("status"));
        assertTrue(report.diagnosticGate(TrainingReportDiagnostics.Severity.INFO).passed());
        assertEquals("NONE", report.highestDiagnosticSeverity());
        assertEquals(Boolean.FALSE, report.hasDiagnosticWarnings());
        assertEquals(Boolean.FALSE, report.hasCriticalDiagnostics());
        assertEquals(List.of(), report.diagnosticFindings());
        assertEquals(TrainingReportActionPlan.Status.READY, report.actionPlan().status());
        assertEquals(List.of(), report.recommendations());
        assertEquals(2, report.history().size());
        assertEquals(2, report.epochSnapshots().size());
        assertEquals(1, report.latestEpochSnapshot().orElseThrow().epoch().orElseThrow());
        assertEquals(0.8, report.latestEpochSnapshot()
                .orElseThrow()
                .validationMetric("accuracy")
                .orElseThrow(), 1e-12);
        assertEquals(0.6, report.epochSnapshot(1).orElseThrow().trainLoss().orElseThrow(), 1e-12);
        assertTrue(report.epochSnapshot(12).isEmpty());
        assertEquals(0.6, report.trainLossSeries().latest().orElseThrow(), 1e-12);
        assertEquals(0.75, report.validationLossSeries().latest().orElseThrow(), 1e-12);
        assertEquals(0.15, report.generalizationGapSeries().latest().orElseThrow(), 1e-12);
        assertEquals(1.25, report.validationToTrainLossRatioSeries().latest().orElseThrow(), 1e-12);
        assertEquals(0.8, report.validationMetricSeries("accuracy").latest().orElseThrow(), 1e-12);
        assertFalse(report.trainMetricSeries("accuracy").available());
        assertEquals(0.75, report.seriesBundle().validationLoss().latest().orElseThrow(), 1e-12);
        assertEquals(0.8, report.seriesBundle().validationMetric("accuracy").latest().orElseThrow(), 1e-12);
        assertTrue(report.seriesBundle().hasValidationMetrics());
        assertEquals(2, report.seriesExport().epochCount());
        assertTrue(report.seriesExport().toCsv().contains("validationLoss"));
        assertEquals("metal", report.metadata().get("device"));
        assertThrows(UnsupportedOperationException.class, () -> report.metadata().put("device", "cpu"));
    }

    @Test
    void typedTrainingReportCoercesScalarFieldsThroughSharedParser() {
        TrainingReport report = TrainingReport.of(Map.of(
                "schema", TrainingReportReader.CANONICAL_SCHEMA,
                "epochCount", "3",
                "bestValidationLoss", "0.42",
                "bestValidationEpoch", "2",
                "durationMs", "987"));

        assertTrue(report.canonical());
        assertEquals(3, report.epochCount());
        assertEquals(0.42, report.bestValidationLoss().orElseThrow(), 1e-12);
        assertEquals(2, report.bestValidationEpoch().orElseThrow());
        assertEquals(987L, report.durationMs());

        TrainingReport invalid = TrainingReport.of(Map.of(
                "epochCount", "bad",
                "bestValidationLoss", "NaN",
                "bestValidationEpoch", "bad",
                "durationMs", "bad"));
        assertEquals(0, invalid.epochCount());
        assertTrue(invalid.bestValidationLoss().isEmpty());
        assertTrue(invalid.bestValidationEpoch().isEmpty());
        assertEquals(0L, invalid.durationMs());
    }

    @Test
    void facadeReadsTrainingReportHistory() throws IOException {
        Path reportFile = tempDir.resolve("canonical-report.json");
        TrainingSummary summary = new TrainingSummary(
                1,
                Double.NaN,
                -1,
                0.25,
                null,
                10L,
                Map.of("epochHistory", List.of(Map.of("epoch", 0, "trainLoss", 0.25))));
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T13:14:15Z"))),
                StandardCharsets.UTF_8);

        assertEquals(1, Gollek.DL.trainingReportHistory(reportFile).size());
        assertEquals(1, Gollek.DL.trainingReportEpochSnapshots(reportFile).size());
        assertEquals(0, Gollek.DL.trainingReportLatestEpochSnapshot(reportFile)
                .orElseThrow()
                .epoch()
                .orElseThrow());
        assertEquals(0.25, Gollek.DL.trainingReportEpochSnapshot(reportFile, 0)
                .orElseThrow()
                .trainLoss()
                .orElseThrow(), 1e-12);
        assertEquals(0.25, Gollek.DL.trainingReportTrainLossSeries(reportFile).latest().orElseThrow(), 1e-12);
        assertTrue(Gollek.DL.trainingReportValidationLossSeries(reportFile).latest().isEmpty());
        assertTrue(Gollek.DL.trainingReportGeneralizationGapSeries(reportFile).latest().isEmpty());
        assertTrue(Gollek.DL.trainingReportGradientL2NormSeries(reportFile).latest().isEmpty());
        assertEquals(0.25, Gollek.DL.trainingReportSeriesBundle(reportFile)
                .trainLoss()
                .latest()
                .orElseThrow(), 1e-12);
        assertEquals(1, Gollek.DL.trainingReportSeriesExport(reportFile).epochCount());
        assertTrue(Gollek.DL.trainingReportSeriesExport(reportFile).toCsv().contains("trainLoss"));
        assertEquals(Boolean.TRUE, Gollek.DL.trainingReportHistorySummary(reportFile).get("available"));
        assertEquals(1, Gollek.DL.trainingReportHistoryOverview(reportFile).size());
        assertEquals(0.25, Gollek.DL.trainingReportHistoryOverview(reportFile)
                .trainLoss()
                .latest()
                .orElseThrow(), 1e-12);
        assertEquals("validation.missing", Gollek.DL.trainingReportDiagnostics(reportFile).get(0).get("code"));
        assertEquals("INFO", Gollek.DL.trainingReportDiagnosticsSummary(reportFile).get("highestSeverity"));
        assertEquals(1, Gollek.DL.trainingReportDiagnosticSummary(reportFile).total());
        assertEquals(1, Gollek.DL.trainingReportDiagnosticSummary(reportFile)
                .count(TrainingReportDiagnostics.Severity.INFO));
        assertEquals("FRESH", Gollek.DL.trainingReportDiagnosticProvenance(reportFile).status());
        assertEquals("FRESH", Gollek.DL.trainingReportDiagnosticsProvenance(reportFile).get("status"));
        assertTrue(Gollek.DL.trainingReportDiagnosticGate(
                reportFile,
                TrainingReportDiagnostics.Severity.INFO).passed());
        assertEquals("INFO", Gollek.DL.trainingReport(reportFile).highestDiagnosticSeverity());
        TrainingReportDiagnostics.Finding finding = Gollek.DL.trainingReportDiagnosticFindings(reportFile).get(0);
        assertEquals(TrainingReportDiagnostics.Severity.INFO, finding.severity());
        assertEquals("validation.missing", finding.code());
        assertEquals(1, finding.evidence().get("historyRows"));
        TrainingReportActionPlan actionPlan = Gollek.DL.trainingReportActionPlan(reportFile);
        assertEquals(TrainingReportActionPlan.Status.NEEDS_ATTENTION, actionPlan.status());
        assertEquals("validation.missing", actionPlan.recommendations().get(0).diagnosticCode());
        assertEquals(
                TrainingReportRecommendation.Category.VALIDATION,
                actionPlan.recommendations().get(0).category());
        assertEquals(
                0.25,
                TrainingReportReader.latestTrainLoss(Gollek.DL.readTrainingReport(reportFile)).orElseThrow(),
                1e-12);
    }

    @Test
    void readerBackfillsRunHealthDiagnosticsForStaleReports() {
        Map<String, Object> primaryIssue = Map.of(
                "kind", "training-failure",
                "code", "non-finite-detected",
                "severity", "error",
                "blocking", true,
                "artifact", "trainer",
                "message", "train gradient must be finite, got NaN",
                "action", "inspect data and loss scale before rerunning");
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schema", TrainingReportReader.CANONICAL_SCHEMA);
        report.put("history", List.of());
        report.put("historySummary", Map.of("available", false));
        report.put("runHealth", Map.ofEntries(
                Map.entry("status", "failed"),
                Map.entry("healthy", false),
                Map.entry("gatePassed", false),
                Map.entry("issueDetected", true),
                Map.entry("issueCount", 1),
                Map.entry("blockingIssueDetected", true),
                Map.entry("blockingIssueCount", 1),
                Map.entry("recommendedAction", primaryIssue.get("action")),
                Map.entry("primaryIssue", primaryIssue),
                Map.entry("primaryBlockingIssue", primaryIssue),
                Map.entry("issueCodes", List.of("non-finite-detected")),
                Map.entry("issueSeverities", List.of("error")),
                Map.entry("issueCountsByKind", Map.of("training-failure", 1)),
                Map.entry("issueCountsBySeverity", Map.of("error", 1)),
                Map.entry("issues", List.of(primaryIssue))));
        report.put("diagnostics", List.of(Map.of(
                "severity", "INFO",
                "code", "history.missing",
                "message", "No epoch history is available in this training report.",
                "evidence", Map.of("historyRows", 0))));
        report.put("diagnosticsSummary", Map.of(
                "available", true,
                "total", 1,
                "highestSeverity", "INFO",
                "bySeverity", Map.of("INFO", 1, "WARNING", 0, "CRITICAL", 0),
                "codes", List.of("history.missing")));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportReader.diagnosticFindings(report);

        assertTrue(hasCode(findings, "history.missing"));
        assertTrue(hasCode(findings, "run_health.gate_failed"));
        assertEquals("CRITICAL", TrainingReportReader.diagnosticsSummary(report).get("highestSeverity"));
        assertEquals(2, TrainingReportReader.diagnosticSummary(report).total());
        assertTrue(TrainingReportReader.diagnosticSummary(report).hasCritical());
        TrainingReportDiagnostics.Provenance provenance = TrainingReportReader.diagnosticProvenance(report);
        assertEquals("BACKFILLED", provenance.status());
        assertEquals(false, provenance.fresh());
        assertEquals(true, provenance.stale());
        assertEquals(true, provenance.diagnosticsPersisted());
        assertEquals(true, provenance.diagnosticsSummaryPersisted());
        assertEquals(true, provenance.diagnosticsBackfilled());
        assertEquals(true, provenance.diagnosticsSummaryStale());
        assertEquals(1, provenance.persistedFindingCount());
        assertEquals(2, provenance.effectiveFindingCount());
        assertEquals("INFO", provenance.persistedHighestSeverity());
        assertEquals("CRITICAL", provenance.effectiveHighestSeverity());
        assertEquals(List.of("history.missing"), provenance.persistedCodes());
        assertEquals(List.of("history.missing", "run_health.gate_failed"), provenance.effectiveCodes());
        assertEquals(List.of("run_health.gate_failed"), provenance.backfilledCodes());
        assertEquals("BACKFILLED", TrainingReportReader.diagnosticsProvenance(report).get("status"));
        assertTrue(TrainingReportReader.diagnostics(report).stream()
                .anyMatch(row -> "run_health.gate_failed".equals(row.get("code"))));
        TrainingReport trainingReport = TrainingReport.of(report);
        assertEquals("BACKFILLED", trainingReport.diagnosticProvenance().status());
        assertEquals(List.of("run_health.gate_failed"), trainingReport.diagnosticProvenance().backfilledCodes());
        assertFalse(trainingReport.diagnosticGate(TrainingReportDiagnostics.Severity.WARNING).passed());
        assertEquals(TrainingReportActionPlan.Status.BLOCKED, trainingReport.actionPlan().status());
        assertTrue(trainingReport.actionPlan().hasBlockers());
    }

    @Test
    void backfillsDataHealthViewFromMetadataForOlderReports() {
        Map<String, Object> report = Map.of(
                "schema", TrainingReportReader.CANONICAL_SCHEMA,
                "metadata", Map.ofEntries(
                        Map.entry("dataLoaderPlanHealth.available", true),
                        Map.entry("dataLoaderPlanHealthStatus", "healthy"),
                        Map.entry("dataLoaderPlanHealthHealthy", true),
                        Map.entry("dataLoaderPlanHealthGatePassed", true),
                        Map.entry("dataLoaderPlanHealthIssueDetected", false),
                        Map.entry("dataLoaderPlanHealthIssueCount", 0),
                        Map.entry("dataLoaderPlanHealthWarningCount", 0),
                        Map.entry("dataLoaderPlanHealthErrorCount", 0),
                        Map.entry("dataDistributionHealth.available", true),
                        Map.entry("dataDistributionHealthStatus", "warning"),
                        Map.entry("dataDistributionHealthHealthy", false),
                        Map.entry("dataDistributionHealthGatePassed", true),
                        Map.entry("dataDistributionHealthIssueDetected", true),
                        Map.entry("dataDistributionHealthIssueCount", 1),
                        Map.entry("dataDistributionHealthWarningCount", 1),
                        Map.entry("dataDistributionHealthErrorCount", 0),
                        Map.entry("dataDistributionHealthIssueCodes", List.of("data-distribution-class-imbalance")),
                        Map.entry("dataDistributionHealthIssueSeverities", List.of("warning")),
                        Map.entry("dataDistributionHealthRecommendedActions", List.of("rebalance labels")),
                        Map.entry(
                                "dataDistributionHealthIssues",
                                List.of(Map.of("code", "data-distribution-class-imbalance")))));

        TrainingReportDataHealth health = TrainingReportReader.dataHealthView(report);

        assertEquals("healthy", health.loaderPlan().status());
        assertEquals("warning", health.distribution().status());
        assertTrue(health.available());
        assertFalse(health.healthy());
        assertTrue(health.gatePassed());
        assertEquals(1, health.issueCount());
        assertEquals(List.of("data-distribution-class-imbalance"), health.issueCodes());
        assertEquals("warning", TrainingReport.of(report).dataHealth().distribution().status());
        assertTrue(TrainingReportReader.diagnosticFindings(report).stream()
                .anyMatch(finding -> "data_health.issue_detected".equals(finding.code())));
        assertEquals("BACKFILLED", TrainingReportReader.diagnosticProvenance(report).status());
    }

    @Test
    void rejectsNonObjectJson() throws IOException {
        Path reportFile = tempDir.resolve("bad-report.json");
        Files.writeString(reportFile, "[]", StandardCharsets.UTF_8);

        IOException error = assertThrows(IOException.class, () -> TrainingReportReader.read(reportFile));

        assertTrue(error.getMessage().contains("Expected JSON object training report"));
    }

    @Test
    void rejectsUnsupportedCanonicalSchema() throws IOException {
        Path reportFile = tempDir.resolve("wrong-schema-report.json");
        Files.writeString(reportFile, "{\"schema\":\"other\"}", StandardCharsets.UTF_8);

        IOException error = assertThrows(IOException.class, () -> TrainingReportReader.readCanonical(reportFile));

        assertTrue(error.getMessage().contains("Unsupported training report schema"));
    }

    private static boolean hasCode(List<TrainingReportDiagnostics.Finding> findings, String code) {
        return findings.stream().anyMatch(finding -> code.equals(finding.code()));
    }
}
