package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
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
