package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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

class TrainingReportPromotionGateArtifactManifestTest {
    @TempDir
    Path tempDir;

    @Test
    void writesReadsAndVerifiesPromotionGateArtifactManifest() throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts = writeGateArtifacts("roundtrip");
        Instant generatedAt = Instant.parse("2026-05-21T09:10:11Z");

        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPromotionGateArtifactManifest(artifacts, generatedAt);

        assertTrue(Files.isRegularFile(manifest.manifestFile()));
        assertEquals(64, manifest.manifestSha256().length());
        assertEquals(generatedAt, manifest.generatedAt());
        assertTrue(manifest.passed());
        assertTrue(manifest.promotable());

        String properties = Files.readString(manifest.manifestFile());
        assertTrue(properties.contains("formatVersion=1"));
        assertTrue(properties.contains("artifact.json.file=promotion-gate.json"));
        assertTrue(properties.contains("artifact.markdown.file=promotion-gate.md"));
        assertTrue(properties.contains("artifact.junitXml.file=promotion-gate.junit.xml"));
        assertTrue(properties.contains("gate.passed=true"));
        assertTrue(properties.contains("gate.promotable=true"));

        TrainingReportPromotionGateArtifactManifest.ManifestInspection inspection =
                Gollek.DL.readTrainingReportPromotionGateArtifactManifest(artifacts.directory());
        assertEquals(TrainingReportPromotionGateArtifactManifest.FORMAT_VERSION, inspection.formatVersion());
        assertEquals(generatedAt, inspection.generatedAt());
        assertEquals(manifest.manifestSha256(), inspection.manifestSha256());
        assertEquals("PROMOTE", inspection.decisionStatus());
        assertEquals("candidate", inspection.decisionCandidate().orElseThrow());
        assertEquals(artifacts.jsonFile(), inspection.artifact("json").orElseThrow().file());
        assertEquals(artifacts.jsonSha256(), inspection.artifact("json").orElseThrow().sha256());
        assertEquals(artifacts.markdownFile(), inspection.artifact("markdown").orElseThrow().file());
        assertEquals(artifacts.junitXmlFile(), inspection.artifact("junitXml").orElseThrow().file());

        TrainingReportPromotionGateArtifacts.ArtifactInspection gateArtifacts =
                Gollek.DL.readTrainingReportPromotionGateArtifacts(inspection);
        assertTrue(gateArtifacts.passed());
        assertEquals("candidate", gateArtifacts.decisionCandidate().orElseThrow());

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactManifest(manifest);
        assertTrue(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertTrue(verification.artifactBytesMatch());
        assertNotNull(verification.artifactVerification());
        assertTrue(verification.artifactVerification().junitXmlWellFormed());
        verification.requirePassed();
    }

    @Test
    void detectsArtifactTamperingThroughManifestVerification() throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts = writeGateArtifacts("tampered-artifact");
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPromotionGateArtifactManifest(
                        artifacts,
                        Instant.parse("2026-05-21T09:10:11Z"));

        Files.writeString(
                artifacts.markdownFile(),
                "\nTampered after manifest write.\n",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactManifest(manifest);

        assertFalse(verification.passed());
        assertTrue(verification.manifestSha256Matches());
        assertFalse(verification.artifactBytesMatch());
        assertNotNull(verification.artifactVerification());
        assertFalse(verification.artifactVerification().markdownSha256Matches());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("markdown artifact byte size mismatch")));
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("Markdown checksum mismatch")));
        assertThrows(IllegalStateException.class, verification::requirePassed);
    }

    @Test
    void detectsManifestTamperingWhenExpectedManifestChecksumIsAvailable() throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts = writeGateArtifacts("tampered-manifest");
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPromotionGateArtifactManifest(
                        artifacts,
                        Instant.parse("2026-05-21T09:10:11Z"));

        String content = Files.readString(manifest.manifestFile());
        Files.writeString(
                manifest.manifestFile(),
                content.replace("gate.promotable=true", "gate.promotable=false"),
                StandardCharsets.UTF_8);

        TrainingReportPromotionGateArtifactManifest.ManifestVerification verification =
                Gollek.DL.verifyTrainingReportPromotionGateArtifactManifest(manifest);

        assertFalse(verification.passed());
        assertFalse(verification.manifestSha256Matches());
        assertTrue(verification.artifactBytesMatch());
        assertTrue(verification.failures().stream()
                .anyMatch(failure -> failure.contains("Manifest checksum mismatch")));
    }

    @Test
    void rejectsPathLikeCustomManifestNames() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifactManifest.Options("../manifest.properties"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifactManifest.Options("nested/manifest.properties"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingReportPromotionGateArtifactManifest.Options("/tmp/manifest.properties"));
    }

    @Test
    void rejectsUnsafeArtifactFileNamesInsideManifest() throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts = writeGateArtifacts("unsafe-manifest-path");
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPromotionGateArtifactManifest(
                        artifacts,
                        Instant.parse("2026-05-21T09:10:11Z"));

        String content = Files.readString(manifest.manifestFile());
        Files.writeString(
                manifest.manifestFile(),
                content.replace("artifact.json.file=promotion-gate.json", "artifact.json.file=../promotion-gate.json"),
                StandardCharsets.UTF_8);

        IOException error = assertThrows(
                IOException.class,
                () -> Gollek.DL.readTrainingReportPromotionGateArtifactManifest(artifacts.directory()));
        assertTrue(error.getMessage().contains("unsafe artifact manifest file name for json"));
    }

    @Test
    void rejectsMalformedSha256InsideManifest() throws IOException {
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts = writeGateArtifacts("bad-sha-manifest");
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportPromotionGateArtifactManifest(
                        artifacts,
                        Instant.parse("2026-05-21T09:10:11Z"));

        String content = Files.readString(manifest.manifestFile());
        Files.writeString(
                manifest.manifestFile(),
                content.replace(artifacts.jsonSha256(), "not-a-sha"),
                StandardCharsets.UTF_8);

        IOException error = assertThrows(
                IOException.class,
                () -> Gollek.DL.readTrainingReportPromotionGateArtifactManifest(artifacts.directory()));
        assertTrue(error.getMessage().contains("invalid artifact manifest sha256 for json"));
    }

    private TrainingReportPromotionGateArtifacts.ArtifactBundle writeGateArtifacts(String prefix) throws IOException {
        TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                promotionReports(prefix),
                "baseline",
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy(),
                tempDir.resolve(prefix + "-review-artifacts"));
        return Gollek.DL.writeTrainingReportPromotionGateArtifacts(
                tempDir.resolve(prefix + "-gate-artifacts"),
                result);
    }

    private Map<String, Path> promotionReports(String prefix) throws IOException {
        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", writeReport(
                prefix + "-baseline.json",
                100L,
                List.of(
                        Map.of("epoch", 0, "trainLoss", 0.9, "validationLoss", 1.0),
                        Map.of("epoch", 1, "trainLoss", 0.6, "validationLoss", 0.7))));
        reports.put("candidate", writeReport(
                prefix + "-candidate.json",
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
