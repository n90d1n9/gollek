package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainingReportPromotionGateJUnitXmlTest {
    @TempDir
    Path tempDir;

    @Test
    void rendersPassingPromotionGateAsValidJUnitXml() throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("base&line", writeReport(
                "baseline.xml-safe.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate<good>", writeReport(
                "candidate.xml-safe.json",
                90L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.7, "validationLoss", 0.8),
                        Map.of("epoch", 1, "trainLoss", 0.4, "validationLoss", 0.45))));

        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                reports,
                "base&line",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve("gate"));
        String xml = result.junitXml();

        assertEquals(xml, Gollek.DL.trainingReportPromotionGateJUnitXml(result));
        assertWellFormedXml(xml);
        assertTrue(xml.startsWith("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
        assertTrue(xml.contains("<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\"0\""));
        assertTrue(xml.contains("name=\"base&amp;line -&gt; candidate&lt;good&gt;\""));
        assertTrue(xml.contains("name=\"promotion.baseline\" value=\"base&amp;line\""));
        assertTrue(xml.contains("name=\"promotion.candidate\" value=\"candidate&lt;good&gt;\""));
        assertTrue(xml.contains("name=\"artifacts.verified\" value=\"true\""));
        assertTrue(xml.contains("name=\"sourceReports.verified\" value=\"true\""));
        assertTrue(xml.contains("name=\"sourceReports.count\" value=\"2\""));
        assertTrue(xml.contains("<system-out># Gollek Training Promotion Gate"));
        assertFalse(xml.contains("<failure"));
    }

    @Test
    void rendersHeldPromotionGateAsFailedJUnitTestcase() throws IOException {
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
        String xml = Gollek.DL.trainingReportPromotionGateJUnitXml(result);

        assertWellFormedXml(xml);
        assertTrue(xml.contains("<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\"1\""));
        assertTrue(xml.contains("<failure type=\"HOLD\""));
        assertTrue(xml.contains("Promotion gate failed"));
        assertTrue(xml.contains("name=\"promotion.status\" value=\"HOLD\""));
        assertTrue(xml.contains("**Gate:** `FAIL`"));
    }

    @Test
    void rendersArtifactVerificationFailureAsFailedJUnitTestcase() throws IOException {
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
                tempDir.resolve("tampered"));

        Files.writeString(
                result.artifacts().markdownFile(),
                Files.readString(result.artifacts().markdownFile()) + "\nTampered after promotion.\n",
                StandardCharsets.UTF_8);
        TrainingReportPromotionArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportPromotionArtifacts(result.artifacts());
        TrainingReportPromotionGate.Result tamperedResult =
                new TrainingReportPromotionGate.Result(result.review(), result.artifacts(), verification);
        String xml = tamperedResult.junitXml();

        assertWellFormedXml(xml);
        assertTrue(xml.contains("<failure type=\"ARTIFACT_VERIFICATION\""));
        assertTrue(xml.contains("name=\"promotion.status\" value=\"PROMOTE\""));
        assertTrue(xml.contains("name=\"artifacts.verified\" value=\"false\""));
        assertTrue(xml.contains("Markdown checksum mismatch"));
    }

    @Test
    void rendersSourceVerificationFailureAsFailedJUnitTestcase() throws IOException {
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
        String xml = result.junitXml();

        assertWellFormedXml(xml);
        assertTrue(xml.contains("<failure type=\"SOURCE_REPORT_VERIFICATION\""));
        assertTrue(xml.contains("name=\"artifacts.verified\" value=\"true\""));
        assertTrue(xml.contains("name=\"sourceReports.verified\" value=\"false\""));
        assertTrue(xml.contains("candidate source report SHA-256 mismatch"));
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

    private static void assertWellFormedXml(String xml) {
        assertDoesNotThrow(() -> DocumentBuilderFactory.newInstance()
                .newDocumentBuilder()
                .parse(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8))));
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
