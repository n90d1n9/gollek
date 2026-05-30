package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
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

class TrainingReportPromotionMarkdownTest {
    @TempDir
    Path tempDir;

    @Test
    void rendersPromotionReviewAsCiFriendlyMarkdown() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("base`line", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate|slow", writeReport(
                "candidate-slow.json",
                200L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.6, "validationLoss", 0.5),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.3))));
        reports.put("candidate-clean", writeReport(
                "candidate-clean.json",
                110L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        String markdown = Gollek.DL.trainingReportPromotionMarkdown(
                reports,
                "base`line",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy());

        assertTrue(markdown.startsWith("# Gollek Training Promotion Review\n"));
        assertTrue(markdown.contains("**Decision:** `PROMOTE` candidate `candidate-clean`"));
        assertTrue(markdown.contains("baseline `base\\`line`"));
        assertTrue(markdown.contains("## Source Reports"));
        assertTrue(markdown.contains("| Role | Name | Source | Bytes | SHA-256 |"));
        assertTrue(markdown.contains("`base\\`line`"));
        assertTrue(markdown.contains("`candidate-clean`"));
        assertTrue(markdown.contains(TrainerCheckpointIO.sha256Hex(reports.get("base`line"))));
        assertTrue(markdown.contains(TrainerCheckpointIO.sha256Hex(reports.get("candidate-clean"))));
        assertTrue(markdown.contains(Long.toString(Files.size(reports.get("candidate-clean")))));
        assertTrue(markdown.contains("| Candidate | Status | Validation score | Diagnostics | Improved | Regressed | Reasons |"));
        assertTrue(markdown.contains("| `candidate\\|slow` | `HOLD` | 0.3 | `pass` / `NONE`"));
        assertTrue(markdown.contains("comparison.duration.regressed"));
        assertTrue(markdown.contains("`bestValidationLoss`"));
        assertTrue(markdown.contains("`durationMs`"));
        assertFalse(markdown.contains("null"));
    }

    @Test
    void rendersNoEligibleCandidateDecision() throws IOException {
        Map<String, Path> reports = Map.of("baseline", writeReport(
                "baseline-only.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.6))));

        String markdown = Gollek.DL.trainingReportPromotionMarkdown(
                reports,
                "baseline",
                TrainingReportDiagnostics.Severity.INFO);

        assertTrue(markdown.contains("`NO_ELIGIBLE_CANDIDATE`"));
        assertTrue(markdown.contains("No candidate passed diagnostics at or below INFO"));
        assertTrue(markdown.contains("## Source Reports"));
        assertTrue(markdown.contains("`baseline`"));
        assertTrue(markdown.contains("| Candidate | Status | Validation score | Diagnostics | Improved | Regressed | Reasons |"));
    }

    private Path writeReport(String fileName, long durationMs, List<Map<String, Object>> epochHistory)
            throws IOException {
        Path reportFile = tempDir.resolve(fileName);
        TrainingSummary summary = new TrainingSummary(
                epochHistory.size(),
                bestLoss(epochHistory, "validationLoss"),
                bestEpoch(epochHistory, "validationLoss"),
                latestLoss(epochHistory, "trainLoss"),
                latestLoss(epochHistory, "validationLoss"),
                durationMs,
                Map.of("epochHistory", epochHistory));
        Files.writeString(
                reportFile,
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-20T09:10:11Z"))),
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
