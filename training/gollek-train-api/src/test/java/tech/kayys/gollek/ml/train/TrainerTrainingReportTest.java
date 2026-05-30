package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerTrainingReportTest {

    @Test
    @SuppressWarnings("unchecked")
    void payloadPreservesCanonicalSchemaAndSummaryValues() {
        Map<String, Object> metadata = Map.of(
                "device", "metal",
                "metrics", List.of("accuracy", "f1"));
        TrainingSummary summary = new TrainingSummary(
                4,
                0.125,
                3,
                0.25,
                0.15,
                1234L,
                metadata);

        Map<String, Object> payload = TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T02:03:04Z"));

        assertEquals(TrainerTrainingReport.SCHEMA, payload.get("schema"));
        assertEquals("2026-05-18T02:03:04Z", payload.get("generatedAt"));
        assertEquals(4, payload.get("epochCount"));
        assertEquals(0.125, payload.get("bestValidationLoss"));
        assertEquals(3, payload.get("bestValidationEpoch"));
        assertEquals(0.25, payload.get("latestTrainLoss"));
        assertEquals(0.15, payload.get("latestValidationLoss"));
        assertEquals(1234L, payload.get("durationMs"));
        assertEquals(List.of(), payload.get("history"));
        assertEquals(1, ((List<?>) payload.get("diagnostics")).size());
        Map<String, Object> diagnosticsSummary = (Map<String, Object>) payload.get("diagnosticsSummary");
        assertEquals(1, diagnosticsSummary.get("total"));
        assertEquals("INFO", diagnosticsSummary.get("highestSeverity"));
        assertEquals(Boolean.FALSE, diagnosticsSummary.get("hasWarnings"));
        Map<String, Object> runHealth = (Map<String, Object>) payload.get("runHealth");
        assertEquals("healthy", runHealth.get("status"));
        assertEquals(Boolean.TRUE, runHealth.get("healthy"));
        assertEquals(Boolean.TRUE, runHealth.get("gatePassed"));
        Map<String, Object> dataHealth = (Map<String, Object>) payload.get("dataHealth");
        assertEquals(Boolean.FALSE, dataHealth.get("available"));
        assertEquals(Boolean.TRUE, dataHealth.get("gatePassed"));
        assertEquals(0, dataHealth.get("issueCount"));
        assertSame(metadata, payload.get("metadata"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void payloadPromotesRunHealthIntoTypedReportSection() {
        Map<String, Object> primaryIssue = Map.of(
                "kind", "training-failure",
                "code", "non-finite-detected",
                "severity", "error",
                "blocking", true,
                "artifact", "trainer",
                "message", "train gradient must be finite, got NaN",
                "action", "inspect data and loss scale, reduce learning rate, or enable gradient clipping");
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("trainingRunHealthStatus", "failed"),
                Map.entry("trainingRunHealthy", false),
                Map.entry("trainingRunGatePassed", false),
                Map.entry("trainingRunIssueDetected", true),
                Map.entry("trainingRunIssueCount", 1),
                Map.entry("trainingRunBlockingIssueDetected", true),
                Map.entry("trainingRunBlockingIssueCount", 1),
                Map.entry("trainingRunRecommendedAction", primaryIssue.get("action")),
                Map.entry("trainingRunPrimaryIssueAvailable", true),
                Map.entry("trainingRunPrimaryIssueKind", primaryIssue.get("kind")),
                Map.entry("trainingRunPrimaryIssueCode", primaryIssue.get("code")),
                Map.entry("trainingRunPrimaryIssueSeverity", primaryIssue.get("severity")),
                Map.entry("trainingRunPrimaryIssueBlocking", true),
                Map.entry("trainingRunPrimaryIssueArtifact", primaryIssue.get("artifact")),
                Map.entry("trainingRunPrimaryIssueMessage", primaryIssue.get("message")),
                Map.entry("trainingRunPrimaryIssueRecommendedAction", primaryIssue.get("action")),
                Map.entry("trainingRunPrimaryBlockingIssueAvailable", true),
                Map.entry("trainingRunPrimaryBlockingIssueKind", primaryIssue.get("kind")),
                Map.entry("trainingRunPrimaryBlockingIssueCode", primaryIssue.get("code")),
                Map.entry("trainingRunPrimaryBlockingIssueSeverity", primaryIssue.get("severity")),
                Map.entry("trainingRunPrimaryBlockingIssueBlocking", true),
                Map.entry("trainingRunPrimaryBlockingIssueArtifact", primaryIssue.get("artifact")),
                Map.entry("trainingRunPrimaryBlockingIssueMessage", primaryIssue.get("message")),
                Map.entry("trainingRunPrimaryBlockingIssueRecommendedAction", primaryIssue.get("action")),
                Map.entry("trainingRunIssueCodes", List.of("non-finite-detected")),
                Map.entry("trainingRunIssueSeverities", List.of("error")),
                Map.entry("trainingRunIssueCountsByKind", Map.of("training-failure", 1)),
                Map.entry("trainingRunIssueCountsBySeverity", Map.of("error", 1)),
                Map.entry("trainingRunIssues", List.of(primaryIssue)));
        TrainingSummary summary = new TrainingSummary(
                1,
                Double.NaN,
                -1,
                Double.NaN,
                null,
                10L,
                metadata);

        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T07:08:09Z")));
        TrainingReportRunHealth health = report.runHealth();

        assertEquals("failed", health.status());
        assertEquals(false, health.healthy());
        assertEquals(false, health.gatePassed());
        assertEquals(true, health.blockingIssueDetected());
        assertEquals(1, health.blockingIssueCount());
        assertEquals(List.of("non-finite-detected"), health.issueCodes());
        assertEquals(Map.of("training-failure", 1), health.issueCountsByKind());
        assertEquals("non-finite-detected", health.primaryIssue().get("code"));
        assertEquals("trainer", health.primaryBlockingIssue().get("artifact"));
        assertEquals(primaryIssue.get("action"), report.runHealthMap().get("recommendedAction"));
        assertEquals("non-finite-detected", ((Map<String, Object>) report.runHealthMap()
                .get("primaryIssue")).get("code"));
        assertTrue(report.diagnosticFindings().stream()
                .anyMatch(finding -> "run_health.gate_failed".equals(finding.code())));
        assertTrue(report.hasCriticalDiagnostics());
        assertEquals(TrainingReportActionPlan.Status.BLOCKED, report.actionPlan().status());
        assertTrue(report.actionPlan().hasBlockers());
    }

    @Test
    @SuppressWarnings("unchecked")
    void payloadPromotesDataHealthIntoTypedReportSection() {
        Map<String, Object> loaderIssue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "message", "dropLast discarded 3 training samples",
                "action", "disable dropLast or choose a batch size that divides the training set");
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("dataLoaderPlanHealth.available", true),
                Map.entry("dataLoaderPlanHealthStatus", "warning"),
                Map.entry("dataLoaderPlanHealthHealthy", false),
                Map.entry("dataLoaderPlanHealthGatePassed", true),
                Map.entry("dataLoaderPlanHealthIssueDetected", true),
                Map.entry("dataLoaderPlanHealthIssueCount", 1),
                Map.entry("dataLoaderPlanHealthWarningCount", 1),
                Map.entry("dataLoaderPlanHealthErrorCount", 0),
                Map.entry("dataLoaderPlanHealthIssueCodes", List.of("data-loader-train-drop-last-discarded-samples")),
                Map.entry("dataLoaderPlanHealthIssueSeverities", List.of("warning")),
                Map.entry(
                        "dataLoaderPlanHealthRecommendedActions",
                        List.of("disable dropLast or choose a batch size that divides the training set")),
                Map.entry("dataLoaderPlanHealthIssues", List.of(loaderIssue)),
                Map.entry("dataDistributionHealth.available", true),
                Map.entry("dataDistributionHealthStatus", "healthy"),
                Map.entry("dataDistributionHealthHealthy", true),
                Map.entry("dataDistributionHealthGatePassed", true),
                Map.entry("dataDistributionHealthIssueDetected", false),
                Map.entry("dataDistributionHealthIssueCount", 0),
                Map.entry("dataDistributionHealthWarningCount", 0),
                Map.entry("dataDistributionHealthErrorCount", 0),
                Map.entry("dataDistributionHealthIssueCodes", List.of()),
                Map.entry("dataDistributionHealthIssueSeverities", List.of()),
                Map.entry("dataDistributionHealthRecommendedActions", List.of()),
                Map.entry("dataDistributionHealthIssues", List.of()));
        TrainingSummary summary = new TrainingSummary(
                1,
                0.4,
                0,
                0.5,
                0.4,
                25L,
                metadata);

        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T09:10:11Z")));
        TrainingReportDataHealth health = report.dataHealth();

        assertEquals("warning", health.loaderPlan().status());
        assertEquals("healthy", health.distribution().status());
        assertEquals(1, health.issueCount());
        assertEquals(1, health.warningCount());
        assertEquals(0, health.errorCount());
        assertEquals(Boolean.TRUE, report.dataHealthMap().get("available"));
        assertEquals(Boolean.FALSE, report.dataHealthMap().get("healthy"));
        assertEquals(
                "data-loader-train-drop-last-discarded-samples",
                ((List<String>) report.dataHealthMap().get("issueCodes")).get(0));
        assertEquals(
                "warning",
                ((Map<String, Object>) report.dataHealthMap().get("loaderPlan")).get("status"));
        assertEquals(
                "data-loader-train-drop-last-discarded-samples",
                ((List<Map<String, Object>>) report.dataHealthMap().get("issues")).get(0).get("code"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void payloadPromotesEpochHistoryAndSummarizesLossTrends() {
        Map<String, Object> metadata = Map.of(
                "epochHistory",
                List.of(
                        Map.ofEntries(
                                Map.entry("epoch", 0),
                                Map.entry("trainLoss", 0.8),
                                Map.entry("validationLoss", 0.9),
                                Map.entry("learningRate", 0.1),
                                Map.entry("gradientL2Norm", 0.8),
                                Map.entry("gradientZeroFraction", 0.25),
                                Map.entry("gradientNonFiniteCount", 0L),
                                Map.entry("gradientNonFiniteFraction", 0.0),
                                Map.entry("parameterL2Norm", 4.0),
                                Map.entry("parameterNonFiniteCount", 0L),
                                Map.entry("parameterNonFiniteFraction", 0.0),
                                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                                Map.entry("parameterUpdateL2Norm", 0.04),
                                Map.entry("parameterUpdateToParameterL2Ratio", 0.01),
                                Map.entry("parameterUpdateToGradientL2Ratio", 0.05),
                                Map.entry("parameterUpdateNonFiniteCount", 0L),
                                Map.entry("parameterUpdateNonFiniteFraction", 0.0),
                                Map.entry("trainMetrics", Map.of("mae", 0.4)),
                                Map.entry("validationMetrics", Map.of("accuracy", 0.5))),
                        Map.ofEntries(
                                Map.entry("epoch", 1),
                                Map.entry("trainLoss", 0.5),
                                Map.entry("validationLoss", 0.7),
                                Map.entry("learningRate", 0.05),
                                Map.entry("gradientL2Norm", 0.4),
                                Map.entry("gradientZeroFraction", 0.0),
                                Map.entry("gradientNonFiniteCount", 0L),
                                Map.entry("gradientNonFiniteFraction", 0.0),
                                Map.entry("parameterL2Norm", 4.2),
                                Map.entry("parameterNonFiniteCount", 0L),
                                Map.entry("parameterNonFiniteFraction", 0.0),
                                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                                Map.entry("parameterUpdateL2Norm", 0.08),
                                Map.entry("parameterUpdateToParameterL2Ratio", 0.02),
                                Map.entry("parameterUpdateToGradientL2Ratio", 0.2),
                                Map.entry("parameterUpdateNonFiniteCount", 0L),
                                Map.entry("parameterUpdateNonFiniteFraction", 0.0),
                                Map.entry("trainMetrics", Map.of("mae", 0.2)),
                                Map.entry("validationMetrics", Map.of("accuracy", 0.75)))));
        TrainingSummary summary = new TrainingSummary(
                2,
                0.7,
                1,
                0.5,
                0.7,
                250L,
                metadata);

        Map<String, Object> payload = TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T08:09:10Z"));

        List<Map<String, Object>> history = (List<Map<String, Object>>) payload.get("history");
        assertEquals(2, history.size());
        assertEquals(0, history.get(0).get("epoch"));
        assertEquals(Map.of("mae", 0.4), history.get(0).get("trainMetrics"));

        Map<String, Object> historySummary = (Map<String, Object>) payload.get("historySummary");
        assertEquals(Boolean.TRUE, historySummary.get("available"));
        assertEquals(2, historySummary.get("size"));
        assertEquals(0, historySummary.get("firstEpoch"));
        assertEquals(1, historySummary.get("lastEpoch"));

        Map<String, Object> trainLoss = (Map<String, Object>) historySummary.get("trainLoss");
        assertEquals(Boolean.TRUE, trainLoss.get("available"));
        assertEquals(0.8, (double) trainLoss.get("first"), 1e-12);
        assertEquals(0.5, (double) trainLoss.get("latest"), 1e-12);
        assertEquals(-0.3, (double) trainLoss.get("deltaFromFirst"), 1e-12);
        assertEquals("improved", trainLoss.get("trend"));

        Map<String, Object> validationLoss = (Map<String, Object>) historySummary.get("validationLoss");
        assertEquals(0.7, (double) validationLoss.get("best"), 1e-12);
        assertEquals(1, validationLoss.get("bestEpoch"));
        assertEquals("improved", validationLoss.get("trend"));

        Map<String, Object> learningRate = (Map<String, Object>) historySummary.get("learningRate");
        assertEquals(0.05, (double) learningRate.get("latest"), 1e-12);
        assertEquals("decreased", learningRate.get("trend"));

        Map<String, Object> generalization = (Map<String, Object>) historySummary.get("generalization");
        assertEquals(Boolean.TRUE, generalization.get("available"));
        assertEquals(2, generalization.get("count"));
        assertEquals(0.1, (double) generalization.get("firstGap"), 1e-12);
        assertEquals(0.2, (double) generalization.get("latestGap"), 1e-12);
        assertEquals(0.2, (double) generalization.get("maxGap"), 1e-12);
        assertEquals(1, generalization.get("maxGapEpoch"));
        assertEquals(0.1, (double) generalization.get("gapDeltaFromFirst"), 1e-12);
        assertEquals("increasing", generalization.get("gapTrend"));
        assertEquals(Boolean.TRUE, generalization.get("gapIncreasing"));
        assertEquals(1.4, (double) generalization.get("latestValidationToTrainLossRatio"), 1e-12);
        assertEquals(Boolean.TRUE, generalization.get("latestValidationLossAboveTrainLoss"));

        Map<String, Object> trainMetrics = (Map<String, Object>) historySummary.get("trainMetrics");
        Map<String, Object> mae = (Map<String, Object>) trainMetrics.get("mae");
        assertEquals(Boolean.TRUE, mae.get("available"));
        assertEquals(0.4, (double) mae.get("first"), 1e-12);
        assertEquals(0.2, (double) mae.get("latest"), 1e-12);
        assertEquals(0.2, (double) mae.get("min"), 1e-12);
        assertEquals(0.4, (double) mae.get("max"), 1e-12);
        assertEquals("decreased", mae.get("trend"));

        Map<String, Object> validationMetrics = (Map<String, Object>) historySummary.get("validationMetrics");
        Map<String, Object> accuracy = (Map<String, Object>) validationMetrics.get("accuracy");
        assertEquals(Boolean.TRUE, accuracy.get("available"));
        assertEquals(0.5, (double) accuracy.get("first"), 1e-12);
        assertEquals(0.75, (double) accuracy.get("latest"), 1e-12);
        assertEquals(0.75, (double) accuracy.get("max"), 1e-12);
        assertEquals("increased", accuracy.get("trend"));

        Map<String, Object> optimization = (Map<String, Object>) historySummary.get("optimization");
        assertEquals(Boolean.TRUE, optimization.get("available"));
        assertEquals(2, optimization.get("count"));
        assertEquals(0, optimization.get("firstEpoch"));
        assertEquals(1, optimization.get("lastEpoch"));
        Map<String, Object> gradients = (Map<String, Object>) optimization.get("gradients");
        Map<String, Object> gradientL2Norm = (Map<String, Object>) gradients.get("l2Norm");
        assertEquals(Boolean.TRUE, gradientL2Norm.get("available"));
        assertEquals(0.8, (double) gradientL2Norm.get("first"), 1e-12);
        assertEquals(0.4, (double) gradientL2Norm.get("latest"), 1e-12);
        assertEquals("decreased", gradientL2Norm.get("trend"));
        Map<String, Object> parameterUpdates = (Map<String, Object>) optimization.get("parameterUpdates");
        assertEquals(Boolean.TRUE, parameterUpdates.get("enabled"));
        Map<String, Object> updateRatio = (Map<String, Object>) parameterUpdates.get("toParameterL2Ratio");
        assertEquals(0.02, (double) updateRatio.get("latest"), 1e-12);
        assertEquals("increased", updateRatio.get("trend"));
        Map<String, Object> latestOptimization = (Map<String, Object>) optimization.get("latest");
        assertEquals(Boolean.TRUE, latestOptimization.get("available"));
        assertEquals(1, latestOptimization.get("epoch"));
        assertEquals(0.4, (double) latestOptimization.get("gradientL2Norm"), 1e-12);
        assertEquals(0.02, (double) latestOptimization.get("parameterUpdateToParameterL2Ratio"), 1e-12);

        String json = TrainerJson.toJson(payload);
        assertTrue(json.contains("\"history\":[{\"epoch\":0"));
        assertTrue(json.contains("\"historySummary\""));
        assertTrue(json.contains("\"optimization\""));
        assertTrue(json.contains("\"validationMetrics\":{\"accuracy\""));
        assertTrue(json.contains("\"trend\":\"improved\""));
    }

    @Test
    void payloadSerializesWithStableSchemaField() {
        TrainingSummary summary = new TrainingSummary(
                1,
                Double.NaN,
                -1,
                0.5,
                null,
                10L,
                Map.of("checkpointManifestSaved", true));

        String json = TrainerJson.toJson(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-18T05:06:07Z")));

        assertEquals(
                "{\"bestValidationEpoch\":-1,\"bestValidationLoss\":null,"
                        + "\"dataHealth\":{\"available\":false,"
                        + "\"distribution\":{\"available\":false,\"errorCount\":0,"
                        + "\"gatePassed\":true,\"healthy\":false,\"issueCodes\":[],"
                        + "\"issueCount\":0,\"issueDetected\":false,"
                        + "\"issueSeverities\":[],\"issues\":[],"
                        + "\"recommendedActions\":[],"
                        + "\"skipReason\":\"data-distribution-health-metadata-missing\","
                        + "\"status\":\"unknown\",\"warningCount\":0},"
                        + "\"errorCount\":0,\"gatePassed\":true,\"healthy\":false,"
                        + "\"issueCodes\":[],\"issueCount\":0,"
                        + "\"issueDetected\":false,\"issueSeverities\":[],"
                        + "\"issues\":[],"
                        + "\"loaderPlan\":{\"available\":false,\"errorCount\":0,"
                        + "\"gatePassed\":true,\"healthy\":false,\"issueCodes\":[],"
                        + "\"issueCount\":0,\"issueDetected\":false,"
                        + "\"issueSeverities\":[],\"issues\":[],"
                        + "\"recommendedActions\":[],"
                        + "\"skipReason\":\"data-loader-plan-health-metadata-missing\","
                        + "\"status\":\"unknown\",\"warningCount\":0},"
                        + "\"recommendedActions\":[],\"warningCount\":0},"
                        + "\"diagnostics\":[{\"code\":\"history.missing\","
                        + "\"evidence\":{\"historyRows\":0},"
                        + "\"message\":\"No epoch history is available in this training report.\","
                        + "\"severity\":\"INFO\"}],"
                        + "\"diagnosticsSummary\":{\"available\":true,"
                        + "\"bySeverity\":{\"CRITICAL\":0,\"INFO\":1,\"WARNING\":0},"
                        + "\"codes\":[\"history.missing\"],\"hasCritical\":false,"
                        + "\"hasInfo\":true,\"hasWarnings\":false,"
                        + "\"highestSeverity\":\"INFO\",\"total\":1},"
                        + "\"durationMs\":10,"
                        + "\"epochCount\":1,\"generatedAt\":\"2026-05-18T05:06:07Z\","
                        + "\"history\":[],\"historySummary\":{\"available\":false,"
                        + "\"generalization\":{\"available\":false,\"count\":0},"
                        + "\"learningRate\":{\"available\":false,\"count\":0},"
                        + "\"optimization\":{\"available\":false,\"count\":0,"
                        + "\"gradients\":{},\"latest\":{\"available\":false},"
                        + "\"parameterUpdates\":{\"enabled\":false},\"parameters\":{}},"
                        + "\"size\":0,"
                        + "\"trainLoss\":{\"available\":false,\"count\":0},"
                        + "\"trainMetrics\":{},"
                        + "\"validationLoss\":{\"available\":false,\"count\":0},"
                        + "\"validationMetrics\":{}},"
                        + "\"latestTrainLoss\":0.5,\"latestValidationLoss\":null,"
                        + "\"metadata\":{\"checkpointManifestSaved\":true},"
                        + "\"runHealth\":{\"blockingIssueCount\":0,"
                        + "\"blockingIssueDetected\":false,\"gatePassed\":true,"
                        + "\"healthy\":true,\"issueCodes\":[],\"issueCount\":0,"
                        + "\"issueCountsByKind\":{},\"issueCountsBySeverity\":{},"
                        + "\"issueDetected\":false,\"issueSeverities\":[],"
                        + "\"issues\":[],\"primaryBlockingIssue\":{},"
                        + "\"primaryIssue\":{},"
                        + "\"recommendedAction\":\"continue monitoring training\","
                        + "\"status\":\"healthy\"},"
                        + "\"schema\":\"gollek.canonical-trainer.report.v1\"}",
                json);
    }
}
