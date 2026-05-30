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

class TrainingReportDiagnosticsTest {
    @TempDir
    Path tempDir;

    @Test
    void detectsValidationRegressionAndOverfitRisk() throws IOException {
        Path reportFile = writeReport(List.of(
                Map.of("epoch", 0, "trainLoss", 0.5, "validationLoss", 0.55),
                Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.85)));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(reportFile);

        assertTrue(hasCode(findings, "validation.loss_worsened"));
        assertTrue(hasCode(findings, "generalization.overfit_risk"));
        TrainingReportDiagnostics.Finding overfitRisk = finding(findings, "generalization.overfit_risk");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, overfitRisk.severity());
        assertEquals(2.125, (double) overfitRisk.evidence().get("latestValidationToTrainLossRatio"), 1e-12);
        assertEquals(0.4, (double) overfitRisk.evidence().get("gapDeltaFromFirst"), 1e-12);

        Map<String, Object> summary = TrainingReportDiagnostics.summary(findings);
        assertEquals(2, summary.get("total"));
        assertEquals("WARNING", summary.get("highestSeverity"));
        assertEquals(Boolean.TRUE, summary.get("hasWarnings"));

        TrainingReportDiagnostics.Summary typedSummary = TrainingReportDiagnostics.summarize(findings);
        assertEquals(2, typedSummary.total());
        assertEquals(2, typedSummary.count(TrainingReportDiagnostics.Severity.WARNING));
        assertEquals("WARNING", typedSummary.highestSeverity());
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, typedSummary.highestSeverityValue().orElseThrow());
        assertFalse(typedSummary.passes(TrainingReportDiagnostics.Severity.INFO));
        assertTrue(typedSummary.passes(TrainingReportDiagnostics.Severity.WARNING));

        TrainingReportDiagnostics.GateResult gate = TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.INFO);
        assertFalse(gate.passed());
        assertEquals(List.of("validation.loss_worsened", "generalization.overfit_risk"), gate.failingCodes());
        assertTrue(gate.message().contains("failed"));
        assertThrows(IllegalStateException.class, gate::requirePassed);
    }

    @Test
    void detectsMissingValidationAndTrainLossPlateau() throws IOException {
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0),
                Map.of("epoch", 1, "trainLoss", 0.995),
                Map.of("epoch", 2, "trainLoss", 0.994)));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "validation.missing"));
        assertTrue(hasCode(findings, "train.loss_plateau"));
        assertEquals(2, TrainingReportDiagnostics.summary(findings).get("total"));
        TrainingReportDiagnostics.Finding plateau = finding(findings, "train.loss_plateau");
        assertEquals(3, plateau.evidence().get("count"));
        assertEquals(0.006, (double) plateau.evidence().get("relativeDeltaFromFirst"), 1e-12);
    }

    @Test
    void exposesDiagnosticsThroughGollekDlFacade() throws IOException {
        Path reportFile = writeReport(List.of(
                Map.of("epoch", 0, "trainLoss", 0.75),
                Map.of("epoch", 1, "trainLoss", 0.75),
                Map.of("epoch", 2, "trainLoss", 0.749)));

        List<TrainingReportDiagnostics.Finding> findings = Gollek.DL.analyzeTrainingReport(reportFile);

        assertTrue(hasCode(findings, "validation.missing"));
        assertTrue(hasCode(findings, "train.loss_plateau"));
        assertEquals("INFO", Gollek.DL.trainingReportDiagnosticsSummary(reportFile).get("highestSeverity"));
        assertEquals(2, Gollek.DL.trainingReportDiagnosticSummary(reportFile).total());
        assertTrue(Gollek.DL.trainingReportDiagnosticGate(
                reportFile,
                TrainingReportDiagnostics.Severity.INFO).passed());
    }

    @Test
    void returnsNoFindingsForImprovingTrainAndValidationCurves() {
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.2),
                Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.9),
                Map.of("epoch", 2, "trainLoss", 0.45, "validationLoss", 0.55)));

        assertFalse(hasCode(TrainingReportDiagnostics.analyze(report), "validation.loss_worsened"));
        assertFalse(hasCode(TrainingReportDiagnostics.analyze(report), "generalization.overfit_risk"));
        assertTrue(TrainingReportDiagnostics.analyze(report).isEmpty());
    }

    @Test
    void promotesRunHealthGateFailureToCriticalDiagnostic() {
        Map<String, Object> primaryIssue = Map.of(
                "kind", "training-failure",
                "code", "non-finite-detected",
                "severity", "error",
                "blocking", true,
                "artifact", "trainer",
                "message", "train gradient must be finite, got NaN",
                "action", "inspect data and loss scale before rerunning");
        Map<String, Object> report = new LinkedHashMap<>();
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

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        TrainingReportDiagnostics.Finding health = finding(findings, "run_health.gate_failed");
        assertEquals(TrainingReportDiagnostics.Severity.CRITICAL, health.severity());
        assertEquals("failed", health.evidence().get("status"));
        assertEquals("non-finite-detected", health.evidence().get("primaryIssueCode"));
        assertEquals("trainer", health.evidence().get("primaryIssueArtifact"));
        assertTrue(hasCode(findings, "history.missing"));
        assertFalse(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.WARNING).passed());
        TrainingReportActionPlan actionPlan = TrainingReportAdvisor.actionPlan(findings);
        assertEquals(TrainingReportActionPlan.Status.BLOCKED, actionPlan.status());
        assertTrue(actionPlan.hasBlockers());
        assertEquals(TrainingReportRecommendation.Category.RUN_HEALTH,
                actionPlan.blockers().get(0).category());
    }

    @Test
    void promotesNonBlockingRunHealthIssueToWarningDiagnostic() {
        Map<String, Object> primaryIssue = Map.of(
                "kind", "resume-warning",
                "code", "checkpoint-metadata-missing",
                "severity", "warning",
                "blocking", false,
                "artifact", "checkpoint",
                "message", "checkpoint has no trainer runtime metadata",
                "action", "rerun once with runtime checkpoint metadata enabled");
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.2),
                Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.8),
                Map.of("epoch", 2, "trainLoss", 0.4, "validationLoss", 0.5)));
        report.put("runHealth", Map.ofEntries(
                Map.entry("status", "warning"),
                Map.entry("healthy", false),
                Map.entry("gatePassed", true),
                Map.entry("issueDetected", true),
                Map.entry("issueCount", 1),
                Map.entry("blockingIssueDetected", false),
                Map.entry("blockingIssueCount", 0),
                Map.entry("recommendedAction", primaryIssue.get("action")),
                Map.entry("primaryIssue", primaryIssue),
                Map.entry("issueCodes", List.of("checkpoint-metadata-missing")),
                Map.entry("issueSeverities", List.of("warning")),
                Map.entry("issueCountsByKind", Map.of("resume-warning", 1)),
                Map.entry("issueCountsBySeverity", Map.of("warning", 1)),
                Map.entry("issues", List.of(primaryIssue))));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        TrainingReportDiagnostics.Finding health = finding(findings, "run_health.issue_detected");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, health.severity());
        assertEquals("checkpoint-metadata-missing", health.evidence().get("primaryIssueCode"));
        assertFalse(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.INFO).passed());
        assertTrue(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.WARNING).passed());
        assertEquals(TrainingReportActionPlan.Status.NEEDS_ATTENTION,
                TrainingReportAdvisor.actionPlan(findings).status());
    }

    @Test
    void promotesDataHealthWarningsToActionableDiagnostic() {
        Map<String, Object> dataIssue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "message", "dropLast discarded 3 training samples",
                "action", "disable dropLast or choose a batch size that divides the training set");
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.1),
                Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.8),
                Map.of("epoch", 2, "trainLoss", 0.5, "validationLoss", 0.6)));
        report.put("dataHealth", Map.of(
                "loaderPlan", Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "warning"),
                        Map.entry("healthy", false),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", true),
                        Map.entry("issueCount", 1),
                        Map.entry("warningCount", 1),
                        Map.entry("errorCount", 0),
                        Map.entry("issueCodes", List.of("data-loader-train-drop-last-discarded-samples")),
                        Map.entry("issueSeverities", List.of("warning")),
                        Map.entry("recommendedActions", List.of(dataIssue.get("action"))),
                        Map.entry("issues", List.of(dataIssue))),
                "distribution", Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "healthy"),
                        Map.entry("healthy", true),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", false),
                        Map.entry("issueCount", 0),
                        Map.entry("warningCount", 0),
                        Map.entry("errorCount", 0),
                        Map.entry("issues", List.of()))));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        TrainingReportDiagnostics.Finding dataHealth = finding(findings, "data_health.issue_detected");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, dataHealth.severity());
        assertEquals("data-loader-train-drop-last-discarded-samples", dataHealth.evidence().get("primaryIssueCode"));
        assertEquals(1, dataHealth.evidence().get("issueCount"));
        assertFalse(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.INFO).passed());
        assertTrue(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.WARNING).passed());
        TrainingReportActionPlan actionPlan = TrainingReportAdvisor.actionPlan(findings);
        assertEquals(TrainingReportActionPlan.Status.NEEDS_ATTENTION, actionPlan.status());
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH,
                actionPlan.recommendations().get(0).category());
    }

    @Test
    void promotesDataHealthGateFailureToCriticalDiagnostic() {
        Map<String, Object> dataIssue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-plan-unavailable",
                "severity", "error",
                "blocking", true,
                "message", "train loader plan was not captured",
                "action", "enable data-loader plan capture before promotion");
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.1),
                Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.8),
                Map.of("epoch", 2, "trainLoss", 0.5, "validationLoss", 0.6)));
        report.put("dataHealth", Map.of(
                "loaderPlan", Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "error"),
                        Map.entry("healthy", false),
                        Map.entry("gatePassed", false),
                        Map.entry("issueDetected", true),
                        Map.entry("issueCount", 1),
                        Map.entry("warningCount", 0),
                        Map.entry("errorCount", 1),
                        Map.entry("issueCodes", List.of("data-loader-train-plan-unavailable")),
                        Map.entry("issueSeverities", List.of("error")),
                        Map.entry("recommendedActions", List.of(dataIssue.get("action"))),
                        Map.entry("issues", List.of(dataIssue))),
                "distribution", Map.ofEntries(
                        Map.entry("available", true),
                        Map.entry("status", "healthy"),
                        Map.entry("healthy", true),
                        Map.entry("gatePassed", true),
                        Map.entry("issueDetected", false),
                        Map.entry("issueCount", 0),
                        Map.entry("warningCount", 0),
                        Map.entry("errorCount", 0),
                        Map.entry("issues", List.of()))));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        TrainingReportDiagnostics.Finding dataHealth = finding(findings, "data_health.gate_failed");
        assertEquals(TrainingReportDiagnostics.Severity.CRITICAL, dataHealth.severity());
        assertEquals(Boolean.FALSE, dataHealth.evidence().get("gatePassed"));
        assertEquals("data-loader-train-plan-unavailable", dataHealth.evidence().get("primaryIssueCode"));
        assertFalse(TrainingReportDiagnostics.gateFindings(
                findings,
                TrainingReportDiagnostics.Severity.WARNING).passed());
        TrainingReportActionPlan actionPlan = TrainingReportAdvisor.actionPlan(findings);
        assertEquals(TrainingReportActionPlan.Status.BLOCKED, actionPlan.status());
        assertTrue(actionPlan.hasBlockers());
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH,
                actionPlan.blockers().get(0).category());
    }

    @Test
    void detectsNonFiniteOptimizerTelemetry() {
        Map<String, Object> row = optimizerRow(0);
        row.put("gradientNonFiniteCount", 2L);
        row.put("gradientNanCount", 1L);
        row.put("gradientPositiveInfinityCount", 1L);
        row.put("gradientNonFiniteFraction", 0.5);
        Map<String, Object> report = report(List.of(row));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "optimization.non_finite_values"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "optimization.non_finite_values");
        assertEquals(TrainingReportDiagnostics.Severity.CRITICAL, finding.severity());
        assertEquals(List.of(0), finding.evidence().get("epochs"));
        assertEquals(2L, finding.evidence().get("gradientNonFiniteCount"));
        assertEquals(1L, finding.evidence().get("gradientNanCount"));
        assertEquals(1L, finding.evidence().get("gradientPositiveInfinityCount"));
        assertEquals(0.5, (double) finding.evidence().get("gradientMaxNonFiniteFraction"), 1e-12);
        assertTrue(TrainingReport.of(report).hasCriticalDiagnostics());
    }

    @Test
    void detectsZeroGradientAtLatestOptimizerStep() {
        Map<String, Object> row = optimizerRow(0);
        row.put("gradientValueCount", 4L);
        row.put("gradientZeroCount", 4L);
        row.put("gradientZeroFraction", 1.0);
        row.put("gradientL2Norm", 0.0);
        Map<String, Object> report = report(List.of(row));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "optimization.zero_gradient"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "optimization.zero_gradient");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, finding.severity());
        assertEquals(4L, finding.evidence().get("gradientValueCount"));
        assertEquals(1.0, (double) finding.evidence().get("gradientZeroFraction"), 1e-12);
    }

    @Test
    void detectsMissingParameterUpdateWhenGradientExists() {
        Map<String, Object> row = optimizerRow(0);
        row.put("gradientL2Norm", 0.25);
        row.put("gradientZeroFraction", 0.0);
        row.put("parameterUpdateDiagnosticsEnabled", true);
        row.put("parameterUpdateL2Norm", 0.0);
        row.put("parameterUpdateToParameterL2Ratio", 0.0);
        Map<String, Object> report = report(List.of(row));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "optimization.no_parameter_update"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "optimization.no_parameter_update");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, finding.severity());
        assertEquals(0.25, (double) finding.evidence().get("gradientL2Norm"), 1e-12);
        assertEquals(0.0, (double) finding.evidence().get("parameterUpdateL2Norm"), 1e-12);
    }

    @Test
    void detectsOversizedParameterUpdate() {
        Map<String, Object> row = optimizerRow(0);
        row.put("parameterUpdateDiagnosticsEnabled", true);
        row.put("parameterUpdateL2Norm", 2.0);
        row.put("parameterUpdateToParameterL2Ratio", 1.25);
        Map<String, Object> report = report(List.of(row));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "optimization.update_too_large"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "optimization.update_too_large");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, finding.severity());
        assertEquals(1.25, (double) finding.evidence().get("parameterUpdateToParameterL2Ratio"), 1e-12);
        assertEquals(1.0, (double) finding.evidence().get("maxUpdateToParameterRatio"), 1e-12);
    }

    @Test
    void detectsEffectivelyZeroLatestLearningRate() {
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.0, "learningRate", 0.0)));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "learning_rate.too_small"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "learning_rate.too_small");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, finding.severity());
        assertEquals(0.0, (double) finding.evidence().get("latest"), 1e-12);
        assertEquals(0, finding.evidence().get("latestEpoch"));
        assertEquals(TrainingReportActionPlan.Status.NEEDS_ATTENTION, TrainingReport.of(report).actionPlan().status());
        assertEquals(
                TrainingReportRecommendation.Category.LEARNING_RATE,
                TrainingReport.of(report).recommendations().get(0).category());
    }

    @Test
    void detectsSharpLearningRateSpike() {
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.1, "learningRate", 0.001),
                Map.of("epoch", 1, "trainLoss", 0.8, "validationLoss", 0.9, "learningRate", 0.02)));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "learning_rate.spiked"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "learning_rate.spiked");
        assertEquals(TrainingReportDiagnostics.Severity.WARNING, finding.severity());
        assertEquals(0, finding.evidence().get("previousEpoch"));
        assertEquals(1, finding.evidence().get("currentEpoch"));
        assertEquals(20.0, (double) finding.evidence().get("ratio"), 1e-12);
    }

    @Test
    void detectsFlatLearningRateWhenTrainLossPlateaus() {
        Map<String, Object> report = report(List.of(
                Map.of("epoch", 0, "trainLoss", 1.0, "learningRate", 0.01),
                Map.of("epoch", 1, "trainLoss", 0.995, "learningRate", 0.01),
                Map.of("epoch", 2, "trainLoss", 0.994, "learningRate", 0.01)));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "train.loss_plateau"));
        assertTrue(hasCode(findings, "learning_rate.flat_with_train_plateau"));
        TrainingReportDiagnostics.Finding finding = finding(findings, "learning_rate.flat_with_train_plateau");
        assertEquals(TrainingReportDiagnostics.Severity.INFO, finding.severity());
        assertEquals(3, finding.evidence().get("count"));
        assertEquals(0.01, (double) finding.evidence().get("latest"), 1e-12);
    }

    @Test
    void diagnosticsCoerceStringReportTelemetryThroughSharedParser() {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("history", List.of(
                Map.of(
                        "epoch", "0",
                        "learningRate", "0.001",
                        "gradientValueCount", "4",
                        "gradientL2Norm", "0.1",
                        "gradientZeroFraction", "0.0",
                        "gradientNonFiniteCount", "0",
                        "parameterUpdateDiagnosticsEnabled", "true",
                        "parameterUpdateL2Norm", "0.1",
                        "parameterUpdateToParameterL2Ratio", "0.01"),
                Map.ofEntries(
                        Map.entry("epoch", "1"),
                        Map.entry("learningRate", "0.02"),
                        Map.entry("gradientValueCount", "4"),
                        Map.entry("gradientL2Norm", "0.1"),
                        Map.entry("gradientZeroFraction", "0.0"),
                        Map.entry("gradientNonFiniteCount", "2"),
                        Map.entry("gradientNanCount", "1"),
                        Map.entry("gradientPositiveInfinityCount", "1"),
                        Map.entry("gradientNegativeInfinityCount", "0"),
                        Map.entry("gradientNonFiniteFraction", "0.5"),
                        Map.entry("parameterUpdateDiagnosticsEnabled", "true"),
                        Map.entry("parameterUpdateL2Norm", "2.0"),
                        Map.entry("parameterUpdateToParameterL2Ratio", "1.25"))));
        report.put("historySummary", Map.of(
                "trainLoss", Map.of(
                        "available", "true",
                        "count", "3",
                        "first", "1.0",
                        "latest", "0.999",
                        "deltaFromFirst", "-0.001"),
                "validationLoss", Map.of(
                        "available", "true",
                        "trend", "worsened",
                        "first", "0.8",
                        "latest", "1.2",
                        "deltaFromFirst", "0.4"),
                "generalization", Map.of(
                        "available", "true",
                        "gapIncreasing", "true",
                        "latestValidationToTrainLossRatio", "2.0",
                        "gapDeltaFromFirst", "0.25"),
                "learningRate", Map.of(
                        "available", "true",
                        "count", "3",
                        "latest", "0.0",
                        "latestEpoch", "2",
                        "relativeDeltaFromFirst", "0.0")));

        List<TrainingReportDiagnostics.Finding> findings = TrainingReportDiagnostics.analyze(report);

        assertTrue(hasCode(findings, "validation.loss_worsened"));
        assertTrue(hasCode(findings, "generalization.overfit_risk"));
        assertTrue(hasCode(findings, "train.loss_plateau"));
        assertTrue(hasCode(findings, "learning_rate.too_small"));
        assertTrue(hasCode(findings, "learning_rate.spiked"));
        assertTrue(hasCode(findings, "optimization.non_finite_values"));
        assertTrue(hasCode(findings, "optimization.update_too_large"));
        assertEquals(2L, finding(findings, "optimization.non_finite_values")
                .evidence()
                .get("gradientNonFiniteCount"));
        assertEquals(1.25, (double) finding(findings, "optimization.update_too_large")
                .evidence()
                .get("parameterUpdateToParameterL2Ratio"), 1e-12);
    }

    private Path writeReport(List<Map<String, Object>> epochHistory) throws IOException {
        Path reportFile = tempDir.resolve("canonical-report.json");
        Files.writeString(reportFile, TrainerJson.toJson(report(epochHistory)), StandardCharsets.UTF_8);
        return reportFile;
    }

    private static Map<String, Object> report(List<Map<String, Object>> epochHistory) {
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                Double.NaN,
                -1,
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                100L,
                Map.of("epochHistory", epochHistory));
        return TrainerTrainingReport.payload(summary, Instant.parse("2026-05-18T16:17:18Z"));
    }

    private static Map<String, Object> optimizerRow(int epoch) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("epoch", epoch);
        row.put("trainLoss", 1.0);
        row.put("gradientL2Norm", 0.1);
        row.put("gradientValueCount", 4L);
        row.put("gradientZeroCount", 0L);
        row.put("gradientZeroFraction", 0.0);
        row.put("gradientNonFiniteCount", 0L);
        row.put("gradientNanCount", 0L);
        row.put("gradientPositiveInfinityCount", 0L);
        row.put("gradientNegativeInfinityCount", 0L);
        row.put("gradientNonFiniteFraction", 0.0);
        row.put("parameterNonFiniteCount", 0L);
        row.put("parameterNanCount", 0L);
        row.put("parameterPositiveInfinityCount", 0L);
        row.put("parameterNegativeInfinityCount", 0L);
        row.put("parameterNonFiniteFraction", 0.0);
        row.put("parameterUpdateDiagnosticsEnabled", false);
        row.put("parameterUpdateL2Norm", 0.0);
        row.put("parameterUpdateNonFiniteCount", 0L);
        row.put("parameterUpdateNanCount", 0L);
        row.put("parameterUpdatePositiveInfinityCount", 0L);
        row.put("parameterUpdateNegativeInfinityCount", 0L);
        row.put("parameterUpdateNonFiniteFraction", 0.0);
        row.put("parameterUpdateToParameterL2Ratio", 0.0);
        return row;
    }

    private static double latestLoss(List<Map<String, Object>> epochHistory, String key) {
        for (int index = epochHistory.size() - 1; index >= 0; index--) {
            Object value = epochHistory.get(index).get(key);
            if (value instanceof Number number) {
                return number.doubleValue();
            }
        }
        return Double.NaN;
    }

    private static boolean hasCode(List<TrainingReportDiagnostics.Finding> findings, String code) {
        return findings.stream().anyMatch(finding -> code.equals(finding.code()));
    }

    private static TrainingReportDiagnostics.Finding finding(
            List<TrainingReportDiagnostics.Finding> findings,
            String code) {
        return findings.stream()
                .filter(finding -> code.equals(finding.code()))
                .findFirst()
                .orElseThrow();
    }
}
