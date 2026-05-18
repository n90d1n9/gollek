package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerRuntimeArtifactPersistenceTest {
    @TempDir
    Path tempDir;

    @Test
    void persistsHistoryAndRefreshesManifest() throws Exception {
        Path history = tempDir.resolve("history.csv");
        Path manifest = tempDir.resolve("manifest.properties");
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("epoch", 1);
        row.put("trainLoss", 0.5);

        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistHistory(
                        history,
                        List.of(row),
                        manifestRequest(manifest, Map.of("history", history)));

        assertTrue(result.artifact().stateChanged());
        assertTrue(result.artifact().saved());
        assertNull(result.artifact().error());
        assertTrue(result.manifest().saved());
        assertTrue(Files.readString(history).contains("epoch,trainLoss"));
        assertTrue(Files.readString(manifest).contains("artifact.history.file=history.csv"));
    }

    @Test
    void persistsReportAndRefreshesManifest() throws Exception {
        Path report = tempDir.resolve("report.json");
        Path manifest = tempDir.resolve("manifest.properties");
        TrainingSummary summary = new TrainingSummary(
                2,
                0.2,
                1,
                0.3,
                0.2,
                100,
                Map.of("device", "cpu"));

        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistReport(
                        report,
                        summary,
                        manifestRequest(manifest, Map.of("report", report)));

        assertTrue(result.artifact().saved());
        assertTrue(result.manifest().saved());
        assertTrue(Files.readString(report).contains("\"schema\":\"gollek.canonical-trainer.report.v1\""));
        assertTrue(Files.readString(manifest).contains("artifact.report.file=report.json"));
    }

    @Test
    void persistsManifestDirectly() throws Exception {
        Path history = tempDir.resolve("history.csv");
        Path manifest = tempDir.resolve("manifest.properties");
        Files.writeString(history, "epoch\n1\n");

        TrainerRuntimeArtifactPersistence.SaveResult result =
                TrainerRuntimeArtifactPersistence.persistManifest(
                        manifestRequest(manifest, Map.of("history", history)));

        assertTrue(result.saved());
        assertNull(result.error());
        assertTrue(Files.readString(manifest).contains("artifact.history.sha256="));
    }

    @Test
    void skipsManifestRefreshWhenArtifactIsSkipped() {
        Path manifest = tempDir.resolve("manifest.properties");

        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistHistory(
                        null,
                        List.of(),
                        manifestRequest(manifest, Map.of()));

        assertFalse(result.artifact().stateChanged());
        assertFalse(result.artifact().saved());
        assertFalse(result.manifest().stateChanged());
        assertFalse(Files.exists(manifest));
    }

    @Test
    void skipsManifestRefreshWhenArtifactWriteFails() throws Exception {
        Path manifest = tempDir.resolve("manifest.properties");
        Path nonDirectoryParent = tempDir.resolve("not-a-dir");
        Files.writeString(nonDirectoryParent, "not a directory");
        Path badHistory = nonDirectoryParent.resolve("history.csv");

        TrainerRuntimeArtifactPersistence.ArtifactResult result =
                TrainerRuntimeArtifactPersistence.persistHistory(
                        badHistory,
                        List.of(Map.of("epoch", 1)),
                        manifestRequest(manifest, Map.of("history", badHistory)));

        assertTrue(result.artifact().stateChanged());
        assertFalse(result.artifact().saved());
        assertNotNull(result.artifact().error());
        assertFalse(result.manifest().stateChanged());
        assertFalse(Files.exists(manifest));
    }

    private static TrainerRuntimeArtifactPersistence.ManifestRequest manifestRequest(
            Path manifest,
            Map<String, Path> artifacts) {
        return new TrainerRuntimeArtifactPersistence.ManifestRequest(manifest, artifacts, 1);
    }
}
