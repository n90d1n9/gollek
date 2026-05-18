package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class TrainerModelCheckpointMetadataTest {

    private static final TrainerModelCheckpointMetadata.ExpectedModel EXPECTED_MODEL =
            new TrainerModelCheckpointMetadata.ExpectedModel("ExpectedModel", "weight:[2]:2;", 2L);

    @Test
    void compatibilityAcceptsMissingOptionalMetadata() {
        assertNull(TrainerModelCheckpointMetadata.compatibilityMismatch(
                new Properties(),
                null,
                EXPECTED_MODEL,
                1));
    }

    @Test
    void compatibilityAcceptsMatchingMetadataAndIntegrity() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-metadata-match", "abc");
        Properties metadata = matchingMetadata();
        metadata.setProperty("modelCheckpointBytes", "3");
        metadata.setProperty(
                "modelCheckpointSha256",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        assertNull(TrainerModelCheckpointMetadata.compatibilityMismatch(metadata, checkpoint, EXPECTED_MODEL, 1));
    }

    @Test
    void checkReportsMissingMetadataAsCompatible() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-metadata-missing", "abc");
        Path metadata = checkpoint.resolveSibling(checkpoint.getFileName() + ".missing.properties");

        TrainerModelCheckpointMetadata.CompatibilityCheck check = TrainerModelCheckpointMetadata.check(
                metadata,
                checkpoint,
                EXPECTED_MODEL,
                1);

        assertTrue(check.report().compatible());
        assertTrue(check.missing());
        assertFalse(check.loaded());
        assertNull(check.loadError());
    }

    @Test
    void writeAndCheckRoundTripIncludesCheckpointIntegrity() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-metadata-roundtrip", "abc");
        Path metadata = Files.createTempFile("gollek-model-metadata-roundtrip", ".properties");
        Files.delete(metadata);

        TrainerModelCheckpointMetadata.write(metadata, checkpoint, EXPECTED_MODEL, 1);
        TrainerModelCheckpointMetadata.CompatibilityCheck check = TrainerModelCheckpointMetadata.check(
                metadata,
                checkpoint,
                EXPECTED_MODEL,
                1);

        assertTrue(check.report().compatible());
        assertTrue(check.loaded());
        assertFalse(check.missing());
        assertNull(check.loadError());

        Properties saved = new Properties();
        try (var reader = Files.newBufferedReader(metadata, StandardCharsets.UTF_8)) {
            saved.load(reader);
        }
        assertEquals("3", saved.getProperty("modelCheckpointBytes"));
        assertEquals(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                saved.getProperty("modelCheckpointSha256"));
    }

    @Test
    void compatibilityRejectsUnsupportedAndInvalidMetadataVersion() {
        Properties unsupported = matchingMetadata();
        unsupported.setProperty("formatVersion", "2");
        Properties invalid = matchingMetadata();
        invalid.setProperty("formatVersion", "bad");

        assertEquals(
                "unsupported model metadata format version 2 (supported: 1)",
                TrainerModelCheckpointMetadata.compatibilityMismatch(unsupported, null, EXPECTED_MODEL, 1));
        assertEquals(
                "invalid model metadata format version: bad",
                TrainerModelCheckpointMetadata.compatibilityMismatch(invalid, null, EXPECTED_MODEL, 1));
    }

    @Test
    void compatibilityRejectsCheckpointIntegrityMismatch() throws Exception {
        Path checkpoint = writeTempFile("gollek-model-metadata-integrity", "abc");
        Properties metadata = matchingMetadata();
        metadata.setProperty("modelCheckpointBytes", "4");

        assertEquals(
                "model checkpoint size mismatch (expected 4 bytes, got 3 bytes)",
                TrainerModelCheckpointMetadata.compatibilityMismatch(metadata, checkpoint, EXPECTED_MODEL, 1));
    }

    @Test
    void compatibilityRejectsModelClassMismatch() {
        Properties metadata = matchingMetadata();
        metadata.setProperty("modelClass", "LoadedModel");

        assertEquals(
                "model class mismatch (expected ExpectedModel, got LoadedModel)",
                TrainerModelCheckpointMetadata.compatibilityMismatch(metadata, null, EXPECTED_MODEL, 1));
    }

    @Test
    void compatibilityRejectsParameterSignatureMismatch() {
        Properties metadata = matchingMetadata();
        metadata.setProperty("modelParameterSignature", "different:[1]:1;");

        assertEquals(
                "model parameter signature mismatch",
                TrainerModelCheckpointMetadata.compatibilityMismatch(metadata, null, EXPECTED_MODEL, 1));
    }

    @Test
    void compatibilityRejectsParameterCountMismatch() {
        Properties metadata = matchingMetadata();
        metadata.setProperty("modelParameterCount", "3");

        assertEquals(
                "model parameter count mismatch (expected 2, got 3)",
                TrainerModelCheckpointMetadata.compatibilityMismatch(metadata, null, EXPECTED_MODEL, 1));
    }

    private static Properties matchingMetadata() {
        Properties metadata = new Properties();
        metadata.setProperty("formatVersion", "1");
        metadata.setProperty("modelClass", "ExpectedModel");
        metadata.setProperty("modelParameterSignature", "weight:[2]:2;");
        metadata.setProperty("modelParameterCount", "2");
        return metadata;
    }

    private static Path writeTempFile(String prefix, String content) throws Exception {
        Path path = Files.createTempFile(prefix, ".safetensors");
        Files.writeString(path, content, StandardCharsets.UTF_8);
        return path;
    }
}
