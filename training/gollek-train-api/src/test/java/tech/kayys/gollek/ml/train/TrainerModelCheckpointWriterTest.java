package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerModelCheckpointWriterTest {
    @TempDir
    Path tempDir;

    @Test
    void writesModelCheckpointAtomically() throws Exception {
        Path checkpoint = tempDir.resolve("model.safetensors");

        TrainerModelCheckpointWriter.WriteResult result =
                TrainerModelCheckpointWriter.writeModel(checkpoint, path -> Files.write(path, new byte[] {1, 2, 3}));

        assertTrue(result.written());
        assertNull(result.error());
        assertTrue(Files.isRegularFile(checkpoint));
        assertEquals(3, Files.size(checkpoint));
        assertFalse(Files.exists(checkpoint.resolveSibling("model.safetensors.tmp")));
    }

    @Test
    void writesModelMetadataWithCheckpointIntegrity() throws Exception {
        Path checkpoint = tempDir.resolve("model.safetensors");
        Path metadata = tempDir.resolve("model.metadata");
        Files.write(checkpoint, new byte[] {1, 2, 3});

        TrainerModelCheckpointWriter.WriteResult result = TrainerModelCheckpointWriter.writeMetadata(
                metadata,
                checkpoint,
                new TrainerModelCheckpointMetadata.ExpectedModel("Model", "weight:[3]:3;", 3),
                1);

        assertTrue(result.written());
        assertNull(result.error());
        Properties saved = new Properties();
        try (var reader = Files.newBufferedReader(metadata)) {
            saved.load(reader);
        }
        assertEquals("1", saved.getProperty("formatVersion"));
        assertEquals("Model", saved.getProperty("modelClass"));
        assertEquals("3", saved.getProperty("modelParameterCount"));
        assertEquals("weight:[3]:3;", saved.getProperty("modelParameterSignature"));
        assertEquals("3", saved.getProperty("modelCheckpointBytes"));
        assertNotNull(saved.getProperty("modelCheckpointSha256"));
    }

    @Test
    void capturesModelWriteFailure() {
        Path checkpoint = tempDir.resolve("model.safetensors");

        TrainerModelCheckpointWriter.WriteResult result = TrainerModelCheckpointWriter.writeModel(
                checkpoint,
                path -> {
                    throw new IllegalStateException("cannot save model");
                });

        assertFalse(result.written());
        assertEquals("cannot save model", result.error());
        assertFalse(Files.exists(checkpoint));
    }

    @Test
    void skippedWhenTargetIsAbsent() {
        TrainerModelCheckpointWriter.WriteResult result =
                TrainerModelCheckpointWriter.writeModel(null, path -> Files.writeString(path, "ignored"));

        assertFalse(result.written());
        assertNull(result.error());
    }
}
