package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class TrainerCheckpointFileIntegrityTest {

    @Test
    void modelCheckpointMismatchAcceptsMatchingSizeAndSha256() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-checkpoint", "abc");
        Properties metadata = new Properties();
        metadata.setProperty("modelCheckpointBytes", "3");
        metadata.setProperty(
                "modelCheckpointSha256",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        assertNull(TrainerCheckpointFileIntegrity.modelCheckpointMismatch(metadata, checkpoint));
    }

    @Test
    void modelCheckpointMismatchReportsInvalidSizeMetadata() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-checkpoint-invalid-size", "abc");
        Properties metadata = new Properties();
        metadata.setProperty("modelCheckpointBytes", "not-a-number");

        assertEquals(
                "invalid model checkpoint byte size metadata: not-a-number",
                TrainerCheckpointFileIntegrity.modelCheckpointMismatch(metadata, checkpoint));
    }

    @Test
    void modelCheckpointMismatchReportsSizeMismatch() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-checkpoint-size", "abc");
        Properties metadata = new Properties();
        metadata.setProperty("modelCheckpointBytes", "7");

        assertEquals(
                "model checkpoint size mismatch (expected 7 bytes, got 3 bytes)",
                TrainerCheckpointFileIntegrity.modelCheckpointMismatch(metadata, checkpoint));
    }

    @Test
    void modelCheckpointMismatchReportsSha256Mismatch() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-checkpoint-sha", "abc");
        Properties metadata = new Properties();
        metadata.setProperty("modelCheckpointSha256", "deadbeef");

        assertEquals(
                "model checkpoint SHA-256 mismatch (expected deadbeef, got "
                        + "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad)",
                TrainerCheckpointFileIntegrity.modelCheckpointMismatch(metadata, checkpoint));
    }

    @Test
    void manifestArtifactMismatchIgnoresUntrackedArtifacts() throws Exception {
        Path artifact = writeTempFile("gollek-artifact-untracked", "abc");
        Properties manifest = new Properties();
        manifest.setProperty("formatVersion", "1");

        assertNull(TrainerCheckpointFileIntegrity.manifestArtifactMismatch(
                manifest,
                "optimizer",
                artifact,
                1));
    }

    @Test
    void requiredManifestArtifactMismatchRejectsUntrackedArtifacts() throws Exception {
        Path artifact = writeTempFile("gollek-artifact-required-untracked", "abc");
        Properties manifest = new Properties();
        manifest.setProperty("formatVersion", "1");

        assertEquals(
                "optimizer checkpoint is missing from checkpoint manifest",
                TrainerCheckpointFileIntegrity.requiredManifestArtifactMismatch(
                        manifest,
                        "optimizer",
                        artifact,
                        1));
    }

    @Test
    void manifestArtifactMismatchValidatesManifestVersion() throws Exception {
        Path artifact = writeTempFile("gollek-artifact-version", "abc");
        Properties unsupported = new Properties();
        unsupported.setProperty("formatVersion", "2");
        Properties invalid = new Properties();
        invalid.setProperty("formatVersion", "bad");

        assertEquals(
                "unsupported checkpoint manifest format version 2 (supported: 1)",
                TrainerCheckpointFileIntegrity.manifestArtifactMismatch(unsupported, "optimizer", artifact, 1));
        assertEquals(
                "invalid checkpoint manifest format version: bad",
                TrainerCheckpointFileIntegrity.manifestArtifactMismatch(invalid, "optimizer", artifact, 1));
    }

    @Test
    void manifestArtifactMismatchReportsFileNameMismatch() throws Exception {
        Path artifact = writeTempFile("gollek-artifact-file", "abc");
        Properties manifest = new Properties();
        manifest.setProperty("formatVersion", "1");
        manifest.setProperty("artifact.optimizer.file", "different.state");

        assertEquals(
                "optimizer checkpoint file mismatch (expected different.state, got " + artifact.getFileName() + ")",
                TrainerCheckpointFileIntegrity.manifestArtifactMismatch(manifest, "optimizer", artifact, 1));
    }

    @Test
    void manifestArtifactMismatchReportsSizeAndSha256Mismatches() throws Exception {
        Path artifact = writeTempFile("gollek-artifact-integrity", "abc");
        Properties sizeManifest = new Properties();
        sizeManifest.setProperty("formatVersion", "1");
        sizeManifest.setProperty("artifact.optimizer.bytes", "9");
        Properties shaManifest = new Properties();
        shaManifest.setProperty("formatVersion", "1");
        shaManifest.setProperty("artifact.optimizer.sha256", "deadbeef");

        assertEquals(
                "optimizer checkpoint size mismatch (expected 9 bytes, got 3 bytes)",
                TrainerCheckpointFileIntegrity.manifestArtifactMismatch(sizeManifest, "optimizer", artifact, 1));
        assertEquals(
                "optimizer checkpoint SHA-256 mismatch (expected deadbeef, got "
                        + "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad)",
                TrainerCheckpointFileIntegrity.manifestArtifactMismatch(shaManifest, "optimizer", artifact, 1));
    }

    private static Path writeTempFile(String prefix, String content) throws Exception {
        Path path = Files.createTempFile(prefix, ".state");
        Files.writeString(path, content, StandardCharsets.UTF_8);
        return path;
    }
}
