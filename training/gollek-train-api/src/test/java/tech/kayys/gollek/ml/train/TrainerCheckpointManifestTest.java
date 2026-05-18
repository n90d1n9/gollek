package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class TrainerCheckpointManifestTest {

    @Test
    void buildWritesFormatVersionAndGeneratedTime() throws Exception {
        Properties manifest = TrainerCheckpointManifest.build(
                Map.of(),
                7,
                Instant.parse("2026-05-18T00:00:00Z"));

        assertEquals("7", manifest.getProperty("formatVersion"));
        assertEquals("2026-05-18T00:00:00Z", manifest.getProperty("generatedAt"));
    }

    @Test
    void buildIncludesOnlyRegularArtifactsWithFileSizeAndSha256() throws Exception {
        Path dir = Files.createTempDirectory("gollek-checkpoint-manifest");
        Path runtime = writeFile(dir.resolve("canonical-runtime.state"), "abc");
        Path optimizer = writeFile(dir.resolve("canonical-optimizer.state"), "xyz");
        Path nestedDirectory = Files.createDirectory(dir.resolve("not-a-file"));

        Map<String, Path> artifacts = new LinkedHashMap<>();
        artifacts.put("runtime", runtime);
        artifacts.put("optimizer", optimizer);
        artifacts.put("missing", dir.resolve("missing.state"));
        artifacts.put("directory", nestedDirectory);
        artifacts.put("disabled", null);

        Properties manifest = TrainerCheckpointManifest.build(
                artifacts,
                1,
                Instant.parse("2026-05-18T01:02:03Z"));

        assertEquals("canonical-runtime.state", manifest.getProperty("artifact.runtime.file"));
        assertEquals("3", manifest.getProperty("artifact.runtime.bytes"));
        assertEquals(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                manifest.getProperty("artifact.runtime.sha256"));
        assertEquals("canonical-optimizer.state", manifest.getProperty("artifact.optimizer.file"));
        assertEquals("3", manifest.getProperty("artifact.optimizer.bytes"));
        assertEquals(
                "3608bca1e44ea6c4d268eb6db02260269892c0b42b86bbf1e77a6fa16c3c9282",
                manifest.getProperty("artifact.optimizer.sha256"));
        assertNull(manifest.getProperty("artifact.missing.file"));
        assertNull(manifest.getProperty("artifact.directory.file"));
        assertNull(manifest.getProperty("artifact.disabled.file"));
    }

    @Test
    void writeAndCheckArtifactRoundTrip() throws Exception {
        Path dir = Files.createTempDirectory("gollek-checkpoint-manifest-roundtrip");
        Path manifest = dir.resolve("canonical-manifest.properties");
        Path optimizer = writeFile(dir.resolve("canonical-optimizer.state"), "abc");

        TrainerCheckpointManifest.write(
                manifest,
                Map.of("optimizer", optimizer),
                1,
                Instant.parse("2026-05-18T01:02:03Z"));
        TrainerCheckpointManifest.CompatibilityCheck check = TrainerCheckpointManifest.checkArtifact(
                manifest,
                "optimizer",
                optimizer,
                1);

        assertTrue(check.report().compatible());
        assertTrue(check.loaded());
        assertFalse(check.missing());
        assertNull(check.loadError());
    }

    @Test
    void checkArtifactReportsMissingManifestAsCompatible() throws Exception {
        Path dir = Files.createTempDirectory("gollek-checkpoint-manifest-missing");
        Path optimizer = writeFile(dir.resolve("canonical-optimizer.state"), "abc");

        TrainerCheckpointManifest.CompatibilityCheck check = TrainerCheckpointManifest.checkArtifact(
                dir.resolve("missing-manifest.properties"),
                "optimizer",
                optimizer,
                1);

        assertTrue(check.report().compatible());
        assertFalse(check.loaded());
        assertTrue(check.missing());
        assertNull(check.loadError());
    }

    @Test
    void checkArtifactReportsIntegrityMismatch() throws Exception {
        Path dir = Files.createTempDirectory("gollek-checkpoint-manifest-mismatch");
        Path optimizer = writeFile(dir.resolve("canonical-optimizer.state"), "abc");
        Path manifest = dir.resolve("canonical-manifest.properties");
        Properties properties = new Properties();
        properties.setProperty("formatVersion", "1");
        properties.setProperty("artifact.optimizer.bytes", "99");
        try (var writer = Files.newBufferedWriter(manifest, StandardCharsets.UTF_8)) {
            properties.store(writer, "test");
        }

        TrainerCheckpointManifest.CompatibilityCheck check = TrainerCheckpointManifest.checkArtifact(
                manifest,
                "optimizer",
                optimizer,
                1);

        assertFalse(check.report().compatible());
        assertEquals(
                "optimizer checkpoint size mismatch (expected 99 bytes, got 3 bytes)",
                check.report().error());
        assertTrue(check.loaded());
        assertFalse(check.missing());
        assertNull(check.loadError());
    }

    private static Path writeFile(Path path, String content) throws Exception {
        Files.writeString(path, content, StandardCharsets.UTF_8);
        return path;
    }
}
