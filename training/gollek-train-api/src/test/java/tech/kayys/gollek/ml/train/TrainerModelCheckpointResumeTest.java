package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerModelCheckpointResumeTest {
    private static final int METADATA_VERSION = 1;
    private static final TrainerModelCheckpointMetadata.ExpectedModel EXPECTED_MODEL =
            new TrainerModelCheckpointMetadata.ExpectedModel("ExampleModel", "weight:[3]:3;", 3);

    @TempDir
    Path tempDir;

    @Test
    void compatibleCheckpointLoadsModelAndMetadataState() throws Exception {
        Path checkpoint = writeCheckpoint("model");
        Path metadata = tempDir.resolve("canonical-model.metadata");
        TrainerModelCheckpointMetadata.write(metadata, checkpoint, EXPECTED_MODEL, METADATA_VERSION);
        AtomicReference<Path> loaded = new AtomicReference<>();

        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                true,
                checkpoint,
                metadata,
                new AtomicBoolean(false),
                EXPECTED_MODEL,
                METADATA_VERSION,
                true,
                new TrainerCheckpointResumeDiagnostics(),
                loaded::set);

        assertTrue(result.stateChanged());
        assertTrue(result.loaded());
        assertFalse(result.missingOnResume());
        assertFalse(result.compatibilityMismatch());
        assertTrue(result.metadataLoaded());
        assertFalse(result.metadataMissingOnResume());
        assertNull(result.metadataLoadError());
        assertNull(result.loadError());
        assertNull(result.failure());
        assertEquals(checkpoint, loaded.get());
    }

    @Test
    void incompatibleMetadataCanBeReportedWithoutLoadingInLenientMode() throws Exception {
        Path checkpoint = writeCheckpoint("model");
        Path metadata = tempDir.resolve("canonical-model.metadata");
        TrainerModelCheckpointMetadata.write(
                metadata,
                checkpoint,
                new TrainerModelCheckpointMetadata.ExpectedModel("OtherModel", "weight:[3]:3;", 3),
                METADATA_VERSION);
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();
        AtomicBoolean loaderCalled = new AtomicBoolean(false);

        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                true,
                checkpoint,
                metadata,
                new AtomicBoolean(false),
                EXPECTED_MODEL,
                METADATA_VERSION,
                false,
                diagnostics,
                path -> loaderCalled.set(true));

        assertTrue(result.stateChanged());
        assertFalse(result.loaded());
        assertTrue(result.compatibilityMismatch());
        assertTrue(result.metadataLoaded());
        assertEquals(
                "model class mismatch (expected ExampleModel, got OtherModel)",
                result.loadError());
        assertNull(result.failure());
        assertFalse(loaderCalled.get());
        assertEquals(
                "model: model class mismatch (expected ExampleModel, got OtherModel)",
                diagnostics.compatibilityMismatches().getFirst());
    }

    @Test
    void incompatibleMetadataCreatesCanonicalStrictFailure() throws Exception {
        Path checkpoint = writeCheckpoint("model");
        Path metadata = tempDir.resolve("canonical-model.metadata");
        TrainerModelCheckpointMetadata.write(
                metadata,
                checkpoint,
                new TrainerModelCheckpointMetadata.ExpectedModel("ExampleModel", "bias:[1]:1;", 1),
                METADATA_VERSION);

        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                true,
                checkpoint,
                metadata,
                new AtomicBoolean(false),
                EXPECTED_MODEL,
                METADATA_VERSION,
                true,
                new TrainerCheckpointResumeDiagnostics(),
                path -> {
                });

        assertFalse(result.loaded());
        assertTrue(result.compatibilityMismatch());
        assertEquals("model parameter signature mismatch", result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertEquals(
                "Model checkpoint metadata mismatch for resume at "
                        + metadata + ": model parameter signature mismatch",
                result.failure().getMessage());
    }

    @Test
    void loadFailurePreservesCauseInStrictMode() throws Exception {
        Path checkpoint = writeCheckpoint("model");
        Path metadata = tempDir.resolve("canonical-model.metadata");
        TrainerModelCheckpointMetadata.write(metadata, checkpoint, EXPECTED_MODEL, METADATA_VERSION);
        IOException cause = new IOException("cannot deserialize");

        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                true,
                checkpoint,
                metadata,
                new AtomicBoolean(false),
                EXPECTED_MODEL,
                METADATA_VERSION,
                true,
                new TrainerCheckpointResumeDiagnostics(),
                path -> {
                    throw cause;
                });

        assertTrue(result.stateChanged());
        assertFalse(result.loaded());
        assertFalse(result.compatibilityMismatch());
        assertEquals("cannot deserialize", result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertEquals("Failed to load model checkpoint from " + checkpoint, result.failure().getMessage());
        assertSame(cause, result.failure().getCause());
    }

    @Test
    void missingCheckpointUsesCanonicalLoadGatePolicy() {
        Path checkpoint = tempDir.resolve("missing.safetensors");

        TrainerModelCheckpointResume.Result result = TrainerModelCheckpointResume.resume(
                true,
                checkpoint,
                tempDir.resolve("canonical-model.metadata"),
                new AtomicBoolean(false),
                EXPECTED_MODEL,
                METADATA_VERSION,
                true,
                new TrainerCheckpointResumeDiagnostics(),
                path -> {
                });

        assertTrue(result.stateChanged());
        assertTrue(result.missingOnResume());
        assertFalse(result.loaded());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertEquals(
                "Missing model checkpoint artifact for resume: " + checkpoint,
                result.failure().getMessage());
    }

    private Path writeCheckpoint(String content) throws IOException {
        Path checkpoint = tempDir.resolve("canonical-model.safetensors");
        Files.writeString(checkpoint, content);
        return checkpoint;
    }
}
