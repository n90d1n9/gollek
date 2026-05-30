package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportPromotionGateMarkdownTest {
    @TempDir
    Path tempDir;

    @Test
    void rendersPassingGateResultAsCompactMarkdown() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("gate"));

        String markdown = result.markdown();

        assertEquals(markdown, Gollek.DL.trainingReportPromotionGateMarkdown(result));
        assertTrue(markdown.startsWith("# Gollek Training Promotion Gate\n"));
        assertTrue(markdown.contains("**Gate:** `PASS`"));
        assertTrue(markdown.contains("**Decision:** `PROMOTE` candidate `candidate`"));
        assertTrue(markdown.contains("**Artifact verification:** `PASS`"));
        assertTrue(markdown.contains("**Source report verification:** `PASS`"));
        assertTrue(markdown.contains("| JSON review |"));
        assertTrue(markdown.contains("## Source Reports"));
        assertTrue(markdown.contains("| baseline | `baseline` |"));
        assertTrue(markdown.contains(result.artifacts().jsonSha256().substring(0, 12)));
        assertTrue(markdown.contains("Candidates audited: `1`"));
        assertTrue(markdown.contains("Promotable candidates: `1`"));
        assertFalse(markdown.contains("null"));
    }

    @Test
    void rendersVerificationFailuresWhenArtifactsAreTampered() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("tampered"));

        Files.writeString(
                result.artifacts().markdownFile(),
                Files.readString(result.artifacts().markdownFile()) + "\nTampered after gate approval.\n",
                StandardCharsets.UTF_8);

        TrainingReportPromotionArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionArtifacts(result.artifacts());
        TrainingReportPromotionGate.Result tamperedResult =
                new TrainingReportPromotionGate.Result(result.review(), result.artifacts(), verification);
        String markdown = Gollek.DL.trainingReportPromotionGateMarkdown(tamperedResult);

        assertFalse(tamperedResult.passed());
        assertTrue(markdown.contains("**Gate:** `FAIL`"));
        assertTrue(markdown.contains("**Artifact verification:** `FAIL`"));
        assertTrue(markdown.contains("| Markdown review |"));
        assertTrue(markdown.contains("| `no` |"));
        assertTrue(markdown.contains("## Verification Failures"));
        assertTrue(markdown.contains("Markdown checksum mismatch"));
    }

    @Test
    void rendersSourceVerificationFailuresWhenReportsAreTampered() throws IOException {
        Map<String, Path> reports = promotionReports();
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                reports,
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("source-tampered"));

        Files.writeString(
                reports.get("candidate"),
                "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
        String markdown = Gollek.DL.trainingReportPromotionGateMarkdown(result);

        assertFalse(result.passed());
        assertTrue(markdown.contains("**Gate:** `FAIL`"));
        assertTrue(markdown.contains("**Artifact verification:** `PASS`"));
        assertTrue(markdown.contains("**Source report verification:** `FAIL`"));
        assertTrue(markdown.contains("## Source Verification Failures"));
        assertTrue(markdown.contains("candidate source report SHA-256 mismatch"));
    }

    private Map<String, Path> promotionReports() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        return reports;
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
                TrainerJson.toJson(TrainerTrainingReport.payload(summary, Instant.parse("2026-05-21T09:10:11Z"))),
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
