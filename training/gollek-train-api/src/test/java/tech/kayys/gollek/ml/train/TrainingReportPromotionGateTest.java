package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

class TrainingReportPromotionGateTest {
    @TempDir
    Path tempDir;

    @Test
    void runsPromotionGateAndWritesVerifiedArtifacts() throws IOException {
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

        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                reports,
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("gate"));

        assertTrue(result.passed());
        assertTrue(result.promotable());
        assertEquals("candidate", result.decision().candidate().orElseThrow().name());
        assertTrue(result.verification().passed());
        assertTrue(result.sourceVerification().passed());
        assertEquals(2, result.sourceVerification().reports().size());
        assertTrue(Files.isRegularFile(result.artifacts().jsonFile()));
        assertTrue(Files.isRegularFile(result.artifacts().markdownFile()));
        assertTrue(result.message().contains("Promotion gate passed"));
        assertEquals(Boolean.TRUE, result.toMap().get("passed"));
        assertTrue(result.toMap().containsKey("sourceVerification"));
        result.requirePassed();
    }

    @Test
    void failsGateWhenCandidateIsNotPromotableButStillWritesVerifiedArtifacts() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.6),
                        Map.of("epoch", 1, "trainLoss", 0.3, "validationLoss", 0.4))));
        reports.put("candidate", writeReport(
                "candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.7, "validationLoss", 0.85))));

        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                reports,
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("hold"));

        assertFalse(result.passed());
        assertFalse(result.promotable());
        assertEquals(TrainingReportPortfolio.PromotionStatus.HOLD, result.decision().status());
        assertTrue(result.verification().passed());
        assertTrue(Files.isRegularFile(result.artifacts().jsonFile()));
        assertTrue(Files.isRegularFile(result.artifacts().markdownFile()));
        assertTrue(result.message().contains("Promotion gate failed"));
        assertThrows(IllegalStateException.class, result::requirePassed);
    }

    @Test
    void supportsCustomArtifactNamesThroughRequest() throws IOException {
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
        TrainingReportPromotionArtifacts.Options artifactOptions =
                new TrainingReportPromotionArtifacts.Options("ci-promotion.json", "ci-promotion.md");

        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                new TrainingReportPromotionGate.Request(
                        reports,
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("custom"),
                        artifactOptions));

        assertTrue(result.passed());
        assertTrue(result.artifacts().jsonFile().endsWith("ci-promotion.json"));
        assertTrue(result.artifacts().markdownFile().endsWith("ci-promotion.md"));

        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionArtifacts(tempDir.resolve("custom"), artifactOptions);
        assertEquals("PROMOTE", inspection.decisionStatus());
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
    }

    @Test
    void failsGateWhenReviewedSourceReportIsTampered() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "source-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "source-candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
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

        assertFalse(result.passed());
        assertTrue(result.promotable());
        assertTrue(result.verification().passed());
        assertFalse(result.sourceVerification().passed());
        assertTrue(result.sourceVerification().failures().stream()
                .anyMatch(failure -> failure.contains("candidate source report SHA-256 mismatch")));
        assertTrue(result.message().contains("source reports did not verify"));
        assertThrows(IllegalStateException.class, result::requirePassed);
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
