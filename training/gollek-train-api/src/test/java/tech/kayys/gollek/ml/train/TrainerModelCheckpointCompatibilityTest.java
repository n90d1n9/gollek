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
import org.junit.jupiter.api.io.TempDir;

class TrainerModelCheckpointCompatibilityTest {
    private static final TrainerModelCheckpointMetadata.ExpectedModel EXPECTED_MODEL =
            new TrainerModelCheckpointMetadata.ExpectedModel("ExpectedModel", "weight:[3]:3;", 3);

    @TempDir
    Path tempDir;

    @Test
    void reportsCompatibleMetadataState() throws Exception {
        Path checkpoint = writeCheckpoint("abc");
        Path metadata = tempDir.resolve("model.metadata");
        TrainerModelCheckpointMetadata.write(metadata, checkpoint, EXPECTED_MODEL, 1);

        TrainerModelCheckpointCompatibility.Result result =
                TrainerModelCheckpointCompatibility.check(metadata, checkpoint, EXPECTED_MODEL, 1);

        assertTrue(result.report().compatible());
        assertTrue(result.metadataLoaded());
        assertFalse(result.metadataMissing());
        assertNull(result.metadataLoadError());
        assertFalse(result.compatibilityMismatch());
    }

    @Test
    void reportsMissingMetadataAsCompatibleButMissing() throws Exception {
        Path checkpoint = writeCheckpoint("abc");

        TrainerModelCheckpointCompatibility.Result result = TrainerModelCheckpointCompatibility.check(
                tempDir.resolve("missing.metadata"),
                checkpoint,
                EXPECTED_MODEL,
                1);

        assertTrue(result.report().compatible());
        assertFalse(result.metadataLoaded());
        assertTrue(result.metadataMissing());
        assertNull(result.metadataLoadError());
        assertFalse(result.compatibilityMismatch());
    }

    @Test
    void reportsMismatchAndRecordsResumeDiagnostic() throws Exception {
        Path checkpoint = writeCheckpoint("abc");
        Path metadata = tempDir.resolve("model.metadata");
        Properties properties = new Properties();
        properties.setProperty("formatVersion", "1");
        properties.setProperty("modelClass", "LoadedModel");
        properties.setProperty("modelParameterSignature", "weight:[3]:3;");
        properties.setProperty("modelParameterCount", "3");
        try (var writer = Files.newBufferedWriter(metadata, StandardCharsets.UTF_8)) {
            properties.store(writer, "test");
        }

        TrainerModelCheckpointCompatibility.Result result =
                TrainerModelCheckpointCompatibility.check(metadata, checkpoint, EXPECTED_MODEL, 1);
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();
        result.recordMismatch(diagnostics);

        assertFalse(result.report().compatible());
        assertTrue(result.metadataLoaded());
        assertFalse(result.metadataMissing());
        assertTrue(result.compatibilityMismatch());
        assertEquals(
                "model class mismatch (expected ExpectedModel, got LoadedModel)",
                result.report().error());
        assertEquals(
                "model: model class mismatch (expected ExpectedModel, got LoadedModel)",
                diagnostics.compatibilityMismatches().getFirst());
    }

    private Path writeCheckpoint(String content) throws Exception {
        Path checkpoint = tempDir.resolve("model.safetensors");
        Files.writeString(checkpoint, content, StandardCharsets.UTF_8);
        return checkpoint;
    }
}
