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

class TrainingReportPromotionGateArtifactsTest {
    @TempDir
    Path tempDir;

    @Test
    void writesGateJsonMarkdownAndJUnitXmlArtifactsWithChecksums() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("review-artifacts"));

        TrainingReportPromotionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifacts(
                        tempDir.resolve("gate-artifacts"),
                        result);

        assertTrue(bundle.passed());
        assertTrue(bundle.promotable());
        assertEquals("candidate", bundle.result().decision().candidate().orElseThrow().name());
        assertTrue(Files.isRegularFile(bundle.jsonFile()));
        assertTrue(Files.isRegularFile(bundle.markdownFile()));
        assertTrue(Files.isRegularFile(bundle.junitXmlFile()));
        assertEquals(64, bundle.jsonSha256().length());
        assertEquals(64, bundle.markdownSha256().length());
        assertEquals(64, bundle.junitXmlSha256().length());

        String json = Files.readString(bundle.jsonFile());
        assertTrue(json.endsWith("\n"));
        assertTrue(json.contains("\"passed\":true"));
        assertTrue(json.contains("\"decision\""));
        assertTrue(json.contains("\"candidate\":\"candidate\""));

        String markdown = Files.readString(bundle.markdownFile());
        assertTrue(markdown.startsWith("# Gollek Training Promotion Gate\n"));
        assertTrue(markdown.contains("**Gate:** `PASS`"));

        String junitXml = Files.readString(bundle.junitXmlFile());
        assertTrue(junitXml.startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assertTrue(junitXml.contains("<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\"0\""));
        assertTrue(junitXml.contains("name=\"artifacts.verified\" value=\"true\""));

        Map<String, Object> map = bundle.toMap();
        assertEquals(Boolean.TRUE, map.get("passed"));
        assertEquals(bundle.junitXmlFile().toString(), map.get("junitXmlFile"));
        assertEquals(bundle.artifactMap(), map.get("artifact"));
        bundle.requirePassed();

        TrainingReportPromotionGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifacts(bundle.directory());
        assertTrue(inspection.passed());
        assertTrue(inspection.promotable());
        assertEquals("PROMOTE", inspection.decisionStatus());
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
        assertEquals(bundle.jsonSha256(), inspection.jsonSha256());
        assertEquals(bundle.markdownSha256(), inspection.markdownSha256());
        assertEquals(bundle.junitXmlSha256(), inspection.junitXmlSha256());
        assertEquals(bundle.artifactMap(), inspection.artifactMap());

        TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifacts(bundle);
        assertTrue(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        assertTrue(verification.junitXmlSha256Matches());
        assertTrue(verification.junitXmlWellFormed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertEquals(bundle.artifactMap(), verification.artifactMap());
        assertEquals(verification.artifactMap(), verification.toMap().get("artifact"));
        verification.requirePassed();
    }

    @Test
    void supportsCustomSafeGateArtifactNames() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("custom-review-artifacts"));
        TrainingReportPromotionGateArtifacts.Options options =
                new TrainingReportPromotionGateArtifacts.Options(
                        "gate.json",
                        "gate.md",
                        "gate.xml");

        TrainingReportPromotionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifacts(
                        tempDir.resolve("custom-gate-artifacts"),
                        result,
                        options);

        assertTrue(bundle.jsonFile().endsWith("gate.json"));
        assertTrue(bundle.markdownFile().endsWith("gate.md"));
        assertTrue(bundle.junitXmlFile().endsWith("gate.xml"));
        assertTrue(Files.exists(bundle.jsonFile()));
        assertTrue(Files.exists(bundle.markdownFile()));
        assertTrue(Files.exists(bundle.junitXmlFile()));

        TrainingReportPromotionGateArtifacts.ArtifactInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifacts(
                        tempDir.resolve("custom-gate-artifacts"),
                        options);
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
        assertTrue(inspection.junitXml().contains("<testsuite name=\"gollek.training.promotion\""));
    }

    @Test
    void refreshDerivedArtifactsRebuildsReportsFromJsonResult() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("refresh-review-artifacts"));
        TrainingReportPromotionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifacts(
                        tempDir.resolve("refresh-gate-artifacts"),
                        result);

        Files.writeString(
                bundle.markdownFile(),
                "\nmanual stale markdown edit\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);
        Files.writeString(
                bundle.junitXmlFile(),
                "\n<!-- manual stale junit edit -->\n",
                StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifacts.ArtifactVerification stale =
                Gollek.DL.verifyTrainingReportPromotionGateArtifacts(bundle.directory(), null, null, null);
        assertFalse(stale.passed());
        assertFalse(stale.markdownMatchesJson());
        assertFalse(stale.junitXmlMatchesJson());

        TrainingReportPromotionGateArtifacts.ArtifactInspection refreshed =
                Gollek.DL.refreshTrainingReportPromotionGateArtifacts(bundle.directory());
        TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifacts(bundle.directory(), null, null, null);

        assertEquals(bundle.jsonSha256(), refreshed.jsonSha256());
        assertTrue(verification.passed());
        assertTrue(verification.markdownMatchesJson());
        assertTrue(verification.junitXmlMatchesJson());
        assertTrue(!Files.readString(bundle.markdownFile(), StandardCharsets.UTF_8)
                .contains("manual stale markdown edit"));
        assertTrue(!Files.readString(bundle.junitXmlFile(), StandardCharsets.UTF_8)
                .contains("manual stale junit edit"));
    }

    @Test
    void detectsTamperedGateArtifactsAndMalformedJUnitXml() throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("tampered-review-artifacts"));
        TrainingReportPromotionGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportPromotionGateArtifacts(
                        tempDir.resolve("tampered-gate-artifacts"),
                        result);

        Files.writeString(
                bundle.junitXmlFile(),
                "<testsuite><broken></testsuite>",
                StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifacts(bundle);

        assertFalse(verification.passed());
        assertTrue(verification.jsonSha256Matches());
        assertTrue(verification.markdownSha256Matches());
        assertFalse(verification.junitXmlSha256Matches());
        assertFalse(verification.junitXmlWellFormed());
        assertTrue(verification.markdownMatchesJson());
        assertFalse(verification.junitXmlMatchesJson());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML checksum mismatch")));
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML is not well-formed")));
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("JUnit XML report does not match promotion gate JSON")));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void rejectsPathLikeCustomGateArtifactNames() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifacts.Options("../gate.json", "gate.md", "gate.xml"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifacts.Options("gate.json", "nested/gate.md", "gate.xml"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifacts.Options("gate.json", "gate.md", "/tmp/gate.xml"));
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
