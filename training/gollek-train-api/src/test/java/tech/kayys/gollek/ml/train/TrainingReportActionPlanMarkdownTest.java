package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportActionPlanMarkdownTest {
    @TempDir
    Path tempDir;

    @Test
    void rendersBlockingActionPlanForCiOutput() throws IOException {
        Path reportFile = writeReport(
                "blocked-report.json",
                List.of(Map.ofEntries(
                        Map.entry("epoch", 0),
                        Map.entry("trainLoss", 1.0),
                        Map.entry("validationLoss", 1.1),
                        Map.entry("learningRate", 0.0),
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
                        Map.entry("parameterUpdateToParameterL2Ratio", 0.01))));

        TrainingReport report = Gollek.DL.trainingReport(reportFile);
        String markdown = Gollek.DL.trainingReportActionPlanMarkdown(reportFile);

        assertTrue(markdown.startsWith("# Gollek Training Action Plan\n"));
        assertTrue(markdown.contains("**Status:** `BLOCKED`"));
        assertTrue(markdown.contains("| Priority | Category | Diagnostic | Title |"));
        assertTrue(markdown.contains("| `BLOCKER` | `OPTIMIZATION` | `optimization.non_finite_values`"));
        assertTrue(markdown.contains("| `HIGH` | `LEARNING_RATE` | `learning_rate.too_small`"));
        assertTrue(markdown.contains("## Action Items"));
        assertTrue(markdown.contains("gradient clipping"));
        assertTrue(markdown.contains("learning rate"));
        assertFalse(markdown.contains("null"));
        assertTrue(report.actionPlanMarkdown().equals(markdown));
        assertTrue(Gollek.DL.trainingReportActionPlanMarkdown(report).equals(markdown));
        assertTrue(Gollek.DL.trainingReportActionPlanMarkdown(report.actionPlan()).equals(markdown));
    }

    @Test
    void rendersDataHealthSectionWhenReportCarriesDataHealth() {
        Map<String, Object> dataIssue = Map.of(
                "kind", "data-loader-plan",
                "code", "data-loader-train-drop-last-discarded-samples",
                "severity", "warning",
                "message", "dropLast discarded 3 training samples",
                "action", "disable dropLast or choose a batch size that divides the training set");
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("epochHistory", List.of(
                        Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.2, "learningRate", 0.01),
                        Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.9, "learningRate", 0.005),
                        Map.of("epoch", 2, "trainLoss", 0.45, "validationLoss", 0.55, "learningRate", 0.001))),
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
                Map.entry("dataLoaderPlanHealthIssues", List.of(dataIssue)),
                Map.entry("dataDistributionHealth.available", true),
                Map.entry("dataDistributionHealthStatus", "healthy"),
                Map.entry("dataDistributionHealthHealthy", true),
                Map.entry("dataDistributionHealthGatePassed", true),
                Map.entry("dataDistributionHealthIssueDetected", false),
                Map.entry("dataDistributionHealthIssueCount", 0),
                Map.entry("dataDistributionHealthWarningCount", 0),
                Map.entry("dataDistributionHealthErrorCount", 0),
                Map.entry("dataDistributionHealthIssues", List.of()));
        TrainingSummary summary = new TrainingSummary(
                3,
                0.55,
                2,
                0.45,
                0.55,
                100L,
                metadata);
        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));

        String markdown = report.actionPlanMarkdown();

        assertTrue(markdown.contains("**Status:** `NEEDS_ATTENTION`"));
        assertTrue(markdown.contains("## Data Health"));
        assertTrue(markdown.contains("| Loader plan | `yes` | `warning` | `PASS` | 1 | 1 | 0 |"));
        assertTrue(markdown.contains("| Distribution | `yes` | `healthy` | `PASS` | 0 | 0 | 0 |"));
        assertTrue(markdown.contains("`data-loader-train-drop-last-discarded-samples`"));
        assertTrue(markdown.contains("disable dropLast"));
        assertTrue(markdown.contains("| `HIGH` | `DATA_HEALTH` | `data_health.issue_detected`"));
        assertTrue(TrainingReportDataHealthMarkdown.render(report.dataHealth()).contains("## Data Health"));
        assertFalse(markdown.contains("null"));
    }

    @Test
    void rendersRuntimeProfileHotspotsWhenReportCarriesProfilerMetadata() {
        Map<String, Object> metadata = Map.ofEntries(
                Map.entry("epochHistory", List.of(
                        Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.2, "learningRate", 0.01),
                        Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.9, "learningRate", 0.005),
                        Map.of("epoch", 2, "trainLoss", 0.45, "validationLoss", 0.55, "learningRate", 0.001))),
                Map.entry("runtimeProfile.hotspotCount", 2),
                Map.entry("runtimeProfile.primaryHotspot.phase", "train.forward"),
                Map.entry("runtimeProfile.primaryHotspot.totalMillis", 12.5),
                Map.entry("runtimeProfile.primaryHotspot.averageMillis", 6.25),
                Map.entry("runtimeProfile.groupCount", 2),
                Map.entry("runtimeProfile.primaryGroup.name", "train"),
                Map.entry("runtimeProfile.primaryGroup.totalMillis", 16.5),
                Map.entry("runtimeProfile.primaryGroup.averageMillis", 4.125),
                Map.entry("runtimeProfile.totalMillis", 41.5),
                Map.entry("runtimeProfile.balance.bottleneckGroup", "input"),
                Map.entry("runtimeProfile.balance.bottleneck.totalMillis", 21.0),
                Map.entry("runtimeProfile.balance.bottleneck.percentTotal", 50.602409638554214),
                Map.entry("runtimeProfile.balance.input.totalMillis", 21.0),
                Map.entry("runtimeProfile.balance.input.percentTotal", 50.602409638554214),
                Map.entry("runtimeProfile.balance.compute.totalMillis", 20.5),
                Map.entry("runtimeProfile.balance.compute.percentTotal", 49.397590361445786),
                Map.entry("runtimeProfile.balance.train.totalMillis", 16.5),
                Map.entry("runtimeProfile.balance.train.percentTotal", 39.75903614457831),
                Map.entry("runtimeProfile.balance.validation.totalMillis", 0.0),
                Map.entry("runtimeProfile.balance.validation.percentTotal", 0.0),
                Map.entry("runtimeProfile.balance.optimizer.totalMillis", 4.0),
                Map.entry("runtimeProfile.balance.optimizer.percentTotal", 9.63855421686747),
                Map.entry("runtimeProfile.input.train.iterator.count", 2L),
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.count", 6L),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 2.5),
                Map.entry("runtimeProfile.input.train.next.count", 4L),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 12.0),
                Map.entry("runtimeProfile.input.validation.hasNext.count", 3L),
                Map.entry("runtimeProfile.input.validation.hasNext.totalMillis", 1.5),
                Map.entry("runtimeProfile.input.validation.next.count", 2L),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 4.0),
                Map.entry("runtimeProfile.groups", List.of(
                        Map.of(
                                "name", "train",
                                "count", 4L,
                                "totalMillis", 16.5,
                                "averageMillis", 4.125,
                                "maxMillis", 8.0),
                        Map.of(
                                "name", "optimizer",
                                "count", 2L,
                                "totalMillis", 4.0,
                                "averageMillis", 2.0,
                                "maxMillis", 3.0))),
                Map.entry("runtimeProfile.hotspots", List.of(
                        Map.of(
                                "phase", "train.forward",
                                "count", 2L,
                                "totalMillis", 12.5,
                                "averageMillis", 6.25,
                                "maxMillis", 8.0),
                        Map.of(
                                "phase", "optimizer.step",
                                "count", 2L,
                                "totalMillis", 4.0,
                                "averageMillis", 2.0,
                                "maxMillis", 3.0))));
        TrainingSummary summary = new TrainingSummary(
                3,
                0.55,
                2,
                0.45,
                0.55,
                100L,
                metadata);
        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));

        String markdown = report.actionPlanMarkdown();

        assertTrue(report.runtimeProfile().available());
        assertEquals("train", report.runtimeProfile().primaryGroup().orElseThrow().name());
        assertEquals("train.forward", report.runtimeProfile().primaryHotspot().orElseThrow().phase());
        assertTrue(report.runtimeProfile().balance().available());
        assertEquals("input", report.runtimeProfile().balance().bottleneckGroup());
        assertEquals(
                TrainingReportRuntimeProfile.BalanceBucket.INPUT,
                report.runtimeProfile().balance().dominantBucket());
        assertEquals(50.602409638554214,
                report.runtimeProfile().balance().dominantPercent().orElseThrow(), 1e-12);
        assertTrue(report.runtimeProfile().balance().inputBound(50.0));
        assertFalse(report.runtimeProfile().balance().optimizerBound(20.0));
        assertEquals(21.0, report.runtimeProfile().balance().input().totalMillis().orElseThrow(), 1e-12);
        assertEquals(50.602409638554214,
                report.runtimeProfile().balance().input().percentTotal().orElseThrow(), 1e-12);
        Map<?, ?> runtimeBalance = (Map<?, ?>) report.runtimeProfileMap().get("balance");
        Map<?, ?> runtimeBalanceInput = (Map<?, ?>) runtimeBalance.get("input");
        assertEquals(Boolean.TRUE, runtimeBalance.get("available"));
        assertEquals("input", runtimeBalance.get("bottleneckGroup"));
        assertEquals("INPUT", runtimeBalance.get("dominantBucket"));
        assertEquals(50.602409638554214, (double) runtimeBalance.get("dominantPercent"), 1e-12);
        assertEquals(21.0, (double) runtimeBalanceInput.get("totalMillis"), 1e-12);
        assertEquals(runtimeBalance, report.runtimeProfileBalanceMap());
        assertEquals(runtimeBalance, Gollek.DL.trainingReportRuntimeProfileBalanceMap(report));
        assertEquals("input", Gollek.DL.trainingReportRuntimeProfileBalance(report).bottleneckGroup());
        TrainingReportRuntimeProfileBalanceAssessment balanceAssessment =
                report.runtimeProfileBalanceAssessment();
        Map<String, Object> balanceAssessmentMap = report.runtimeProfileBalanceAssessmentMap();
        assertTrue(balanceAssessment.available());
        assertTrue(balanceAssessment.requiresAttention());
        assertTrue(balanceAssessment.inputBound());
        assertFalse(balanceAssessment.optimizerBound());
        assertEquals(
                TrainingReportRuntimeProfile.BalanceBucket.INPUT,
                balanceAssessment.dominantBucket());
        assertEquals(50.602409638554214,
                balanceAssessment.dominantPercent().orElseThrow(), 1e-12);
        assertEquals(Boolean.TRUE, balanceAssessmentMap.get("requiresAttention"));
        assertEquals("INPUT", balanceAssessmentMap.get("dominantBucket"));
        assertEquals(balanceAssessmentMap, Gollek.DL.trainingReportRuntimeProfileBalanceAssessmentMap(report));
        assertEquals(balanceAssessment, Gollek.DL.trainingReportRuntimeProfileBalanceAssessment(report));
        TrainingReportRuntimeProfileBalanceAssessment customBalanceAssessment =
                Gollek.DL.trainingReportRuntimeProfileBalanceAssessment(
                        report,
                        new TrainingReportRuntimeProfileBalanceAssessment.Thresholds(60.0, 45.0, 50.0, 70.0));
        assertFalse(customBalanceAssessment.inputBound());
        assertFalse(customBalanceAssessment.requiresAttention());
        String balanceMarkdown = report.runtimeProfileBalanceMarkdown();
        assertEquals(balanceMarkdown, Gollek.DL.trainingReportRuntimeProfileBalanceMarkdown(report));
        assertTrue(balanceMarkdown.startsWith("### Runtime Balance\n"));
        assertTrue(balanceMarkdown.contains("**Bottleneck group:** `input` (21.000 ms total)"));
        assertEquals("runtime_profile.primary_hotspot", report.actionPlan().recommendations().get(0).diagnosticCode());
        assertEquals(
                TrainingReportRecommendation.Category.TRAINING_DYNAMICS,
                report.actionPlan().recommendations().get(0).category());
        TrainingReportRecommendation groupRecommendation = report.actionPlan().recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group"))
                .findFirst()
                .orElseThrow();
        assertEquals(TrainingReportRecommendation.Category.TRAINING_DYNAMICS, groupRecommendation.category());
        assertEquals("train", groupRecommendation.evidence().get("group"));
        assertEquals(2, groupRecommendation.evidence().get("groupCount"));
        assertTrue(groupRecommendation.actions().stream()
                .anyMatch(action -> action.contains("forward, backward, loss, and metric")));
        Map<String, Object> runtimeInputProfile = report.runtimeInputProfileMap();
        assertEquals(Boolean.TRUE, runtimeInputProfile.get("available"));
        assertEquals(runtimeInputProfile, Gollek.DL.trainingReportRuntimeInputProfile(report));
        Map<?, ?> trainInput = (Map<?, ?>) runtimeInputProfile.get("train");
        Map<?, ?> trainNext = (Map<?, ?>) trainInput.get("next");
        assertEquals(21.0, (double) runtimeInputProfile.get("totalMillis"), 1e-12);
        assertEquals("train", runtimeInputProfile.get("dominantScope"));
        assertEquals(15.5, (double) runtimeInputProfile.get("dominantScopeTotalMillis"), 1e-12);
        assertEquals(15.5 / 21.0 * 100.0, (double) runtimeInputProfile.get("dominantScopePercent"), 1e-12);
        assertEquals(15.5 / 5.5, (double) runtimeInputProfile.get("trainToValidationTotalRatio"), 1e-12);
        assertEquals(15.5, (double) trainInput.get("totalMillis"), 1e-12);
        assertEquals("next", trainInput.get("dominantStage"));
        assertEquals(12.0, (double) trainInput.get("dominantStageTotalMillis"), 1e-12);
        assertEquals(12.0 / 15.5 * 100.0, (double) trainInput.get("dominantStagePercent"), 1e-12);
        assertEquals(4L, trainNext.get("count"));
        assertEquals(12.0, (double) trainNext.get("totalMillis"), 1e-12);
        Map<?, ?> validationInput = (Map<?, ?>) runtimeInputProfile.get("validation");
        Map<?, ?> validationIterator = (Map<?, ?>) validationInput.get("iterator");
        assertEquals(5.5, (double) validationInput.get("totalMillis"), 1e-12);
        assertEquals("next", validationInput.get("dominantStage"));
        assertEquals(4.0, (double) validationInput.get("dominantStageTotalMillis"), 1e-12);
        assertEquals(Boolean.FALSE, validationIterator.get("available"));
        assertTrue(markdown.contains("**Status:** `NEEDS_ATTENTION`"));
        assertTrue(markdown.contains("## Runtime Profile"));
        assertTrue(markdown.contains("**Primary group:** `train` (16.500 ms total)"));
        assertTrue(markdown.contains("### Runtime Balance"));
        assertTrue(markdown.contains("**Bottleneck group:** `input` (21.000 ms total)"));
        assertTrue(markdown.contains("| Bucket | Total ms | Total % |"));
        assertTrue(markdown.contains("| `input` | 21.000 | 50.602 |"));
        assertTrue(markdown.contains("| `compute` | 20.500 | 49.398 |"));
        assertTrue(markdown.contains("### Input Pipeline"));
        assertTrue(markdown.contains(
                "**Input bottleneck:** `train.next()` (12.000 ms of 15.500 ms train input, 77.419% of that scope)"));
        String inputMarkdown = report.runtimeInputProfileMarkdown();
        assertEquals(inputMarkdown, Gollek.DL.trainingReportRuntimeInputProfileMarkdown(report));
        assertTrue(inputMarkdown.startsWith("### Input Pipeline\n"));
        assertFalse(inputMarkdown.contains("## Runtime Profile"));
        assertTrue(inputMarkdown.contains(
                "**Input bottleneck:** `train.next()` (12.000 ms of 15.500 ms train input, 77.419% of that scope)"));
        assertTrue(markdown.contains(
                "| Scope | Iterator count | Iterator ms | HasNext count | HasNext ms | Next count | Next ms | Total ms |"));
        assertTrue(markdown.contains("| `train` | 2 | 1.000 | 6 | 2.500 | 4 | 12.000 | 15.500 |"));
        assertTrue(markdown.contains("| `validation` |  |  | 3 | 1.500 | 2 | 4.000 | 5.500 |"));
        assertTrue(markdown.contains("| `train` | 4 | 16.500 |  | 4.125 |  | 8.000 |  |  |"));
        assertTrue(markdown.contains("| `optimizer` | 2 | 4.000 |  | 2.000 |  | 3.000 |  |  |"));
        assertTrue(markdown.contains("**Primary hotspot:** `train.forward` (12.500 ms total)"));
        assertTrue(markdown.contains("| 1 | `train.forward` | 2 | 12.500 |  | 6.250 |  | 8.000 |  |  |"));
        assertTrue(markdown.contains("| 2 | `optimizer.step` | 2 | 4.000 |  | 2.000 |  | 3.000 |  |  |"));
        assertTrue(markdown.contains("| `MEDIUM` | `TRAINING_DYNAMICS` | `runtime_profile.primary_hotspot`"));
        assertTrue(markdown.contains("| `MEDIUM` | `TRAINING_DYNAMICS` | `runtime_profile.primary_group`"));
        assertTrue(markdown.contains("Enable the fastest available backend"));
        assertTrue(report.runtimeProfileMarkdown().contains("## Runtime Profile"));
        assertTrue(report.runtimeProfileMarkdown().contains("| `train` | 2 | 1.000 | 6 | 2.500 | 4 | 12.000 | 15.500 |"));
        assertFalse(markdown.contains("null"));
    }

    @Test
    void recommendsOptimizerTuningForOptimizerRuntimeHotspots() {
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                Map.ofEntries(
                        Map.entry("epochHistory", List.of(
                                Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8, "learningRate", 0.01))),
                        Map.entry("runtimeProfile.hotspotCount", 1),
                        Map.entry("runtimeProfile.primaryHotspot.phase", "optimizer.step"),
                        Map.entry("runtimeProfile.primaryHotspot.totalMillis", 20.0),
                        Map.entry("runtimeProfile.primaryHotspot.averageMillis", 10.0),
                        Map.entry("runtimeProfile.hotspots", List.of(Map.of(
                                "phase", "optimizer.step",
                                "count", 2L,
                                "totalMillis", 20.0,
                                "averageMillis", 10.0,
                                "maxMillis", 12.0)))));
        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().get(0);

        assertEquals("runtime_profile.primary_hotspot", recommendation.diagnosticCode());
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, recommendation.category());
        assertTrue(recommendation.actions().stream().anyMatch(action -> action.contains("optimizer choice")));
        assertTrue(report.actionPlanMarkdown().contains("| `MEDIUM` | `OPTIMIZATION` | `runtime_profile.primary_hotspot`"));
    }

    @Test
    void recommendsInputPipelineTuningForInputRuntimeHotspots() {
        TrainingReport report = runtimeHotspotReport(
                "input.train.next",
                Map.entry("runtimeProfile.input.train.iterator.count", 2L),
                Map.entry("runtimeProfile.input.train.iterator.totalMillis", 1.0),
                Map.entry("runtimeProfile.input.train.hasNext.count", 6L),
                Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 3.0),
                Map.entry("runtimeProfile.input.train.next.count", 4L),
                Map.entry("runtimeProfile.input.train.next.totalMillis", 20.0),
                Map.entry("runtimeProfile.input.train.next.averageMillis", 5.0),
                Map.entry("runtimeProfile.input.validation.next.count", 2L),
                Map.entry("runtimeProfile.input.validation.next.totalMillis", 4.0));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().get(0);

        assertEquals("runtime_profile.primary_hotspot", recommendation.diagnosticCode());
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH, recommendation.category());
        assertEquals("input.train.next", recommendation.evidence().get("phase"));
        assertEquals(4L, recommendation.evidence().get("trainNextCount"));
        assertEquals(20.0, (double) recommendation.evidence().get("trainNextTotalMillis"), 1e-12);
        assertEquals(24.0, (double) recommendation.evidence().get("trainInputTotalMillis"), 1e-12);
        assertEquals(4.0, (double) recommendation.evidence().get("validationInputTotalMillis"), 1e-12);
        assertEquals(28.0, (double) recommendation.evidence().get("inputTotalMillis"), 1e-12);
        assertEquals("train", recommendation.evidence().get("dominantInputScope"));
        assertEquals(24.0, (double) recommendation.evidence().get("dominantInputScopeTotalMillis"), 1e-12);
        assertEquals(24.0 / 28.0 * 100.0,
                (double) recommendation.evidence().get("dominantInputScopePercent"), 1e-12);
        assertEquals("next", recommendation.evidence().get("dominantInputStage"));
        assertEquals(20.0, (double) recommendation.evidence().get("dominantInputStageTotalMillis"), 1e-12);
        assertEquals(20.0 / 24.0 * 100.0,
                (double) recommendation.evidence().get("dominantInputStagePercent"), 1e-12);
        assertEquals(6.0, (double) recommendation.evidence().get("trainToValidationInputTotalRatio"), 1e-12);
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("Prioritize the `train` input loader `next()` path")
                        && action.contains("20.000 ms of 24.000 ms")));
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("DataLoader.prefetch")));
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("input.train.*") && action.contains("input.validation.*")));
        assertTrue(report.actionPlanMarkdown().contains("| `MEDIUM` | `DATA_HEALTH` | `runtime_profile.primary_hotspot`"));
    }

    @Test
    void recommendsInputPipelineTuningForPrimaryInputRuntimeGroup() {
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                Map.ofEntries(
                        Map.entry("epochHistory", List.of(
                                Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8, "learningRate", 0.01))),
                        Map.entry("runtimeProfile.groupCount", 2),
                        Map.entry("runtimeProfile.primaryGroup.name", "input"),
                        Map.entry("runtimeProfile.primaryGroup.totalMillis", 24.0),
                        Map.entry("runtimeProfile.primaryGroup.averageMillis", 4.0),
                        Map.entry("runtimeProfile.input.train.hasNext.count", 6L),
                        Map.entry("runtimeProfile.input.train.hasNext.totalMillis", 8.0),
                        Map.entry("runtimeProfile.input.train.next.count", 6L),
                        Map.entry("runtimeProfile.input.train.next.totalMillis", 16.0),
                        Map.entry("runtimeProfile.input.validation.next.count", 2L),
                        Map.entry("runtimeProfile.input.validation.next.totalMillis", 5.0),
                        Map.entry("runtimeProfile.groups", List.of(
                                Map.of(
                                        "name", "input",
                                        "count", 6L,
                                        "totalMillis", 24.0,
                                        "averageMillis", 4.0,
                                        "maxMillis", 9.0,
                                        "stddevMillis", 2.0),
                                Map.of(
                                        "name", "train",
                                        "count", 4L,
                                        "totalMillis", 10.0,
                                        "averageMillis", 2.5,
                                        "maxMillis", 3.0)))));
        TrainingReport report = TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));

        TrainingReportRecommendation groupRecommendation = report.actionPlan().recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group"))
                .findFirst()
                .orElseThrow();
        TrainingReportRecommendation variabilityRecommendation = report.actionPlan().recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group_variability"))
                .findFirst()
                .orElseThrow();

        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH, groupRecommendation.category());
        assertEquals("input", groupRecommendation.evidence().get("group"));
        assertEquals(24.0, (double) groupRecommendation.evidence().get("trainInputTotalMillis"), 1e-12);
        assertEquals(5.0, (double) groupRecommendation.evidence().get("validationInputTotalMillis"), 1e-12);
        assertEquals("train", groupRecommendation.evidence().get("dominantInputScope"));
        assertEquals("next", groupRecommendation.evidence().get("dominantInputStage"));
        assertTrue(groupRecommendation.actions().stream()
                .anyMatch(action -> action.contains("Prioritize the `train` input loader `next()` path")
                        && action.contains("16.000 ms of 24.000 ms")));
        assertTrue(groupRecommendation.actions().stream()
                .anyMatch(action -> action.contains("prefetch buffering")));
        assertEquals(TrainingReportRecommendation.Category.DATA_HEALTH, variabilityRecommendation.category());
        assertTrue(variabilityRecommendation.actions().stream()
                .anyMatch(action -> action.contains("loader jitter")));
        assertTrue(report.actionPlanMarkdown().contains("| `MEDIUM` | `DATA_HEALTH` | `runtime_profile.primary_group`"));
        assertTrue(report.actionPlanMarkdown().contains(
                "| `MEDIUM` | `DATA_HEALTH` | `runtime_profile.primary_group_variability`"));
    }

    @Test
    void recommendsSamplingParameterUpdateDiagnosticsWhenParameterScansDominateRuntime() {
        TrainingReport report = runtimeHotspotReport(
                "optimizer.parameterUpdateDiagnostics",
                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                Map.entry("parameterUpdateDiagnosticsSampled", false),
                Map.entry("parameterUpdateDiagnosticsIntervalSteps", 1));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().get(0);

        assertEquals("runtime_profile.primary_hotspot", recommendation.diagnosticCode());
        assertEquals(TrainingReportRecommendation.Category.OPTIMIZATION, recommendation.category());
        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("parameterUpdateDiagnosticsIntervalSteps")));
        assertEquals(Boolean.TRUE, recommendation.evidence().get("parameterUpdateDiagnosticsEnabled"));
        assertEquals(Boolean.FALSE, recommendation.evidence().get("parameterUpdateDiagnosticsSampled"));
        assertEquals(1, recommendation.evidence().get("parameterUpdateDiagnosticsIntervalSteps"));
        assertTrue(report.actionPlanMarkdown().contains("## Parameter Update Diagnostics"));
        assertTrue(report.actionPlanMarkdown().contains("Exact parameter-update diagnostics run every optimizer step."));
    }

    @Test
    void acknowledgesSampledParameterUpdateDiagnosticsWhenParameterScansStillDominateRuntime() {
        TrainingReport report = runtimeHotspotReport(
                "optimizer.parameterUpdateDiagnostics",
                Map.entry("parameterUpdateDiagnosticsEnabled", true),
                Map.entry("parameterUpdateDiagnosticsSampled", true),
                Map.entry("parameterUpdateDiagnosticsIntervalSteps", 8));

        TrainingReportRecommendation recommendation = report.actionPlan().recommendations().get(0);

        assertTrue(recommendation.actions().stream()
                .anyMatch(action -> action.contains("already sampled every 8 optimizer step")));
        assertEquals(Boolean.TRUE, recommendation.evidence().get("parameterUpdateDiagnosticsEnabled"));
        assertEquals(Boolean.TRUE, recommendation.evidence().get("parameterUpdateDiagnosticsSampled"));
        assertEquals(8, recommendation.evidence().get("parameterUpdateDiagnosticsIntervalSteps"));
        assertTrue(report.actionPlanMarkdown().contains("## Parameter Update Diagnostics"));
        assertTrue(report.actionPlanMarkdown().contains("| `yes` | `yes` | 8 |"));
        assertTrue(report.actionPlanMarkdown().contains(
                "Exact parameter-update diagnostics are sampled every 8 optimizer step(s)."));
        assertTrue(report.parameterUpdateDiagnosticsPolicyMarkdown().contains("## Parameter Update Diagnostics"));
    }

    @Test
    void rendersReadyActionPlanWithoutRecommendations() throws IOException {
        Path reportFile = writeReport(
                "ready-report.json",
                List.of(
                        Map.of("epoch", 0, "trainLoss", 1.0, "validationLoss", 1.2, "learningRate", 0.01),
                        Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.9, "learningRate", 0.005),
                        Map.of("epoch", 2, "trainLoss", 0.45, "validationLoss", 0.55, "learningRate", 0.001)));

        String markdown = Gollek.DL.trainingReportActionPlanMarkdown(reportFile);

        assertTrue(markdown.contains("**Status:** `READY`"));
        assertTrue(markdown.contains("No recommendations. The report is ready under the current diagnostics."));
        assertFalse(markdown.contains("## Action Items"));
        assertFalse(markdown.contains("## Parameter Update Diagnostics"));
    }

    private Path writeReport(String fileName, List<Map<String, Object>> epochHistory) throws IOException {
        Path reportFile = tempDir.resolve(fileName);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestLoss(epochHistory, "validationLoss"),
                bestEpoch(epochHistory, "validationLoss"),
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                100L,
                Map.of("epochHistory", epochHistory));
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-26T10:11:12Z"))),
                StandardCharsets.UTF_8);
        return reportFile;
    }

    @SafeVarargs
    private static TrainingReport runtimeHotspotReport(String phase, Map.Entry<String, ?>... metadataEntries) {
        java.util.LinkedHashMap<String, Object> metadata = new java.util.LinkedHashMap<>();
        metadata.put("epochHistory", List.of(
                Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8, "learningRate", 0.01)));
        metadata.put("runtimeProfile.hotspotCount", 1);
        metadata.put("runtimeProfile.primaryHotspot.phase", phase);
        metadata.put("runtimeProfile.primaryHotspot.totalMillis", 20.0);
        metadata.put("runtimeProfile.primaryHotspot.averageMillis", 10.0);
        metadata.put("runtimeProfile.hotspots", List.of(Map.of(
                "phase", phase,
                "count", 2L,
                "totalMillis", 20.0,
                "averageMillis", 10.0,
                "maxMillis", 12.0)));
        for (Map.Entry<String, ?> entry : metadataEntries) {
            metadata.put(entry.getKey(), entry.getValue());
        }
        TrainingSummary summary = new TrainingSummary(
                1,
                0.8,
                0,
                0.7,
                0.8,
                100L,
                metadata);
        return TrainingReport.of(TrainerTrainingReport.payload(
                summary,
                Instant.parse("2026-05-26T11:12:13Z")));
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

    private static double bestLoss(List<Map<String, Object>> epochHistory, String key) {
        double best = Double.POSITIVE_INFINITY;
        for (Map<String, Object> row : epochHistory) {
            Object value = row.get(key);
            if (value instanceof Number number && number.doubleValue() < best) {
                best = number.doubleValue();
            }
        }
        return Double.isFinite(best) ? best : Double.NaN;
    }

    private static int bestEpoch(List<Map<String, Object>> epochHistory, String key) {
        double best = Double.POSITIVE_INFINITY;
        int bestEpoch = -1;
        for (Map<String, Object> row : epochHistory) {
            Object value = row.get(key);
            Object epoch = row.get("epoch");
            if (value instanceof Number number && number.doubleValue() < best) {
                best = number.doubleValue();
                bestEpoch = epoch instanceof Number epochNumber ? epochNumber.intValue() : bestEpoch;
            }
        }
        return bestEpoch;
    }
}
