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

class TrainingReportPromotionArtifactsTest {
    @TempDir
    Path tempDir;

    @Test
    void writesJsonAndMarkdownArtifactsWithChecksums() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                95L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPromotionArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionArtifacts(
                        reports,
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("promotion"));

        assertTrue(bundle.promotable());
        assertEquals("candidate", bundle.decision().candidate().orElseThrow().name());
        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));
        assertEquals(64, bundle.jsonSha256().length());
        assertEquals(64, bundle.markdownSha256().length());

        String json = Files.readString(bundle.jsonFile());
        assertTrue(json.endsWith("\n"));
        assertTrue(json.contains("\"decision\""));
        assertTrue(json.contains("\"candidate\":\"candidate\""));
        assertTrue(json.contains("\"promotable\":true"));
        Map<?, ?> reviewJson = (Map<?, ?>) TrainerJsonParser.parse(json);
        Map<?, ?> baselineReport = (Map<?, ?>) reviewJson.get("baselineReport");
        assertEquals(TrainerCheckpointIO.sha256Hex(reports.get("baseline")), baselineReport.get("sourceSha256"));
        assertEquals(Files.size(reports.get("baseline")), ((Number) baselineReport.get("sourceBytes")).longValue());
        Map<?, ?> candidateReview = (Map<?, ?>) ((List<?>) reviewJson.get("candidates")).get(0);
        Map<?, ?> candidateReport = (Map<?, ?>) candidateReview.get("candidateReport");
        assertEquals(TrainerCheckpointIO.sha256Hex(reports.get("candidate")), candidateReport.get("sourceSha256"));
        Map<?, ?> decision = (Map<?, ?>) reviewJson.get("decision");
        Map<?, ?> decisionCandidateReport = (Map<?, ?>) decision.get("candidateReport");
        assertEquals(
                TrainerCheckpointIO.sha256Hex(reports.get("candidate")),
                decisionCandidateReport.get("sourceSha256"));

        String markdown = Files.readString(bundle.markdownFile());
        assertTrue(markdown.contains("# Gollek Training Promotion Review"));
        assertTrue(markdown.contains("`PROMOTE` candidate `candidate`"));

        Map<String, Object> map = bundle.toMap();
        assertEquals(bundle.jsonFile().toString(), map.get("jsonFile"));
        assertEquals(bundle.markdownFile().toString(), map.get("markdownFile"));
        assertEquals(Boolean.TRUE, map.get("promotable"));

        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionArtifacts(bundle.directory());
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertEquals(bundle.markdownSha256(), inspection.markdownSha256());
        assertTrue(inspection.promotable());
        assertEquals("PROMOTE", inspection.decisionStatus());
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
        assertTrue(inspection.markdown().contains("`PROMOTE` candidate `candidate`"));
        assertEquals(2, inspection.sourceReports().size());
        assertEquals("baseline", inspection.sourceReports().get(0).role());
        assertEquals("candidate", inspection.sourceReports().get(1).name());
        assertTrue(inspection.toMap().containsKey("sourceReports"));

        TrainingReportPromotionArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        verification.requirePassed();

        List<TrainingReportPromotionArtifacts.SourceReport> sourceReports =
                Gollek.DL.trainingReportPromotionSourceReports(inspection);
        assertEquals(2, sourceReports.size());
        assertEquals(TrainerCheckpointIO.sha256Hex(reports.get("baseline")), sourceReports.get(0).sha256());

        TrainingReportPromotionArtifacts.SourceVerification sourceVerification =
                Gollek.DL.verifyTrainingReportPromotionSourceReports(inspection);
        assertTrue(sourceVerification.passed());
        assertEquals(2, sourceVerification.reports().size());
        assertTrue(sourceVerification.toMap().containsKey("reports"));

        TrainingReportPromotionArtifacts.SourceVerification pathSourceVerification =
                Gollek.DL.verifyTrainingReportPromotionSourceReports(bundle.directory());
        assertTrue(pathSourceVerification.passed());
    }

    @Test
    void supportsCustomSafeArtifactNames() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.8, "validationLoss", 0.9),
                        Map.of("epoch", 1, "trainLoss", 0.5, "validationLoss", 0.6))));
        reports.put("candidate", writeReport(
                "candidate.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.75),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.5))));

        TrainingReportPortfolio.PromotionReview review = Gollek.DL.trainingReportPromotionReview(
                reports,
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy());
        TrainingReportPromotionArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionArtifacts(
                        tempDir.resolve("custom"),
                        review,
                        new TrainingReportPromotionArtifacts.Options("review.json", "review.md"));

        assertTrue(bundle.jsonFile().endsWith("review.json"));
        assertTrue(bundle.markdownFile().endsWith("review.md"));
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));

        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionArtifacts(
                        tempDir.resolve("custom"),
                        new TrainingReportPromotionArtifacts.Options("review.json", "review.md"));
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
    }

    @Test
    void detectsTamperedPromotionArtifacts() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                95L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        TrainingReportPromotionArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionArtifacts(
                        reports,
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("tamper"));

        Files.writeString(
                bundle.markdownFile(),
                Files.readString(bundle.markdownFile()) + "\nTampered after approval.\n",
                StandardCharsets.UTF_8);

        TrainingReportPromotionArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionArtifacts(bundle);

        assertFalse(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertFalse(verification.markdownSha256Matches());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown checksum mismatch")));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void detectsTamperedSourceReportRecordedInPromotionReview() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                "baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                "candidate.json",
                95L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));
        TrainingReportPromotionArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionArtifacts(
                        reports,
                        "baseline",
                        TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                        tempDir.resolve("tampered-source"));

        Files.writeString(
                reports.get("candidate"),
                "\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionArtifacts(bundle.directory());
        TrainingReportPromotionArtifacts.SourceVerification verification =
                Gollek.DL.verifyTrainingReportPromotionSourceReports(inspection);

        assertFalse(verification.passed());
        assertEquals(2, verification.reports().size());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("candidate source report SHA-256 mismatch")));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void rejectsPathLikeCustomArtifactNames() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionArtifacts.Options("../review.json", "review.md"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionArtifacts.Options("review.json", "nested/review.md"));
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
