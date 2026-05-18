package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerCheckpointArtifactIntegrityTest {
    @TempDir
    Path tempDir;

    @Test
    void reportsCompatibleArtifactAndManifestState() throws Exception {
        Path artifact = Files.writeString(tempDir.resolve("optimizer.state"), "state");
        Path manifest = tempDir.resolve("manifest.properties");
        TrainerCheckpointManifest.write(
                manifest,
                Map.of("optimizer", artifact),
                1,
                Instant.parse("2026-05-18T00:00:00Z"));

        TrainerCheckpointArtifactIntegrity.Result result =
                TrainerCheckpointArtifactIntegrity.check(manifest, "optimizer", artifact, 1);

        assertTrue(result.report().compatible());
        assertTrue(result.manifestLoaded());
        assertFalse(result.manifestMissing());
        assertNull(result.manifestLoadError());
        assertFalse(result.integrityMismatch());
    }

    @Test
    void reportsMissingManifestAsCompatibleButMissing() throws Exception {
        Path artifact = Files.writeString(tempDir.resolve("scheduler.state"), "state");

        TrainerCheckpointArtifactIntegrity.Result result = TrainerCheckpointArtifactIntegrity.check(
                tempDir.resolve("missing.properties"),
                "scheduler",
                artifact,
                1);

        assertTrue(result.report().compatible());
        assertFalse(result.manifestLoaded());
        assertTrue(result.manifestMissing());
        assertNull(result.manifestLoadError());
        assertFalse(result.integrityMismatch());
    }

    @Test
    void reportsMismatchAndRecordsResumeDiagnostic() throws Exception {
        Path artifact = Files.writeString(tempDir.resolve("history.csv"), "epoch\n1\n");
        Path manifest = tempDir.resolve("manifest.properties");
        Properties properties = new Properties();
        properties.setProperty("formatVersion", "1");
        properties.setProperty("artifact.history.bytes", "999");
        try (var writer = Files.newBufferedWriter(manifest, StandardCharsets.UTF_8)) {
            properties.store(writer, "test");
        }

        TrainerCheckpointArtifactIntegrity.Result result =
                TrainerCheckpointArtifactIntegrity.check(manifest, "history", artifact, 1);
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();
        result.recordMismatch(diagnostics);

        assertFalse(result.report().compatible());
        assertTrue(result.manifestLoaded());
        assertFalse(result.manifestMissing());
        assertTrue(result.integrityMismatch());
        assertEquals(
                "history checkpoint size mismatch (expected 999 bytes, got 8 bytes)",
                result.report().error());
        assertEquals(
                "history: history checkpoint size mismatch (expected 999 bytes, got 8 bytes)",
                diagnostics.compatibilityMismatches().getFirst());
    }
}
