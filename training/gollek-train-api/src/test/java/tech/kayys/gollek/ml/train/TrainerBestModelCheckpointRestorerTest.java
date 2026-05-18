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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerBestModelCheckpointRestorerTest {
    @TempDir
    Path tempDir;

    @Test
    void skipsWhenRestoreIsDisabledOrCheckpointIsMissing() {
        AtomicBoolean integrityChecked = new AtomicBoolean(false);
        AtomicBoolean loaded = new AtomicBoolean(false);

        TrainerBestModelCheckpointRestorer.Result disabled = TrainerBestModelCheckpointRestorer.restore(
                false,
                tempDir.resolve("best.safetensors"),
                tempDir.resolve("manifest.properties"),
                true,
                (name, path) -> {
                    integrityChecked.set(true);
                    return TrainerCheckpointCompatibilityReport.ok();
                },
                path -> loaded.set(true));

        TrainerBestModelCheckpointRestorer.Result missing = TrainerBestModelCheckpointRestorer.restore(
                true,
                tempDir.resolve("missing.safetensors"),
                tempDir.resolve("manifest.properties"),
                true,
                (name, path) -> {
                    integrityChecked.set(true);
                    return TrainerCheckpointCompatibilityReport.ok();
                },
                path -> loaded.set(true));

        assertFalse(disabled.restored());
        assertFalse(missing.restored());
        assertNull(disabled.loadError());
        assertNull(missing.loadError());
        assertFalse(integrityChecked.get());
        assertFalse(loaded.get());
    }

    @Test
    void restoresCompatibleBestModelCheckpoint() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("best.safetensors"), "weights");
        AtomicBoolean loaded = new AtomicBoolean(false);

        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                true,
                checkpoint,
                tempDir.resolve("manifest.properties"),
                true,
                (name, path) -> {
                    assertEquals("bestModel", name);
                    assertEquals(checkpoint, path);
                    return TrainerCheckpointCompatibilityReport.ok();
                },
                path -> {
                    assertEquals(checkpoint, path);
                    loaded.set(true);
                });

        assertTrue(result.restored());
        assertNull(result.loadError());
        assertNull(result.failure());
        assertTrue(loaded.get());
    }

    @Test
    void reportsIntegrityMismatchWithoutFailureWhenFailFastDisabled() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("best.safetensors"), "bad");
        AtomicBoolean loaded = new AtomicBoolean(false);

        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                true,
                checkpoint,
                tempDir.resolve("manifest.properties"),
                false,
                (name, path) -> TrainerCheckpointCompatibilityReport.incompatible("bestModel size mismatch"),
                path -> loaded.set(true));

        assertFalse(result.restored());
        assertEquals("bestModel size mismatch", result.loadError());
        assertNull(result.failure());
        assertFalse(loaded.get());
    }

    @Test
    void createsIntegrityMismatchFailureWhenFailFastEnabled() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("best.safetensors"), "bad");
        Path manifest = tempDir.resolve("manifest.properties");

        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                true,
                checkpoint,
                manifest,
                true,
                (name, path) -> TrainerCheckpointCompatibilityReport.incompatible("bestModel size mismatch"),
                path -> {
                });

        assertFalse(result.restored());
        assertEquals("bestModel size mismatch", result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertTrue(result.failure().getMessage().contains("Best model checkpoint integrity mismatch"));
        assertTrue(result.failure().getMessage().contains(manifest.toString()));
    }

    @Test
    void reportsLoadFailureWithoutFailureWhenFailFastDisabled() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("best.safetensors"), "weights");
        IOException loadFailure = new IOException("cannot decode");

        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                true,
                checkpoint,
                tempDir.resolve("manifest.properties"),
                false,
                (name, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> {
                    throw loadFailure;
                });

        assertFalse(result.restored());
        assertEquals("cannot decode", result.loadError());
        assertNull(result.failure());
    }

    @Test
    void createsLoadFailureWhenFailFastEnabled() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("best.safetensors"), "weights");
        IOException loadFailure = new IOException("cannot decode");

        TrainerBestModelCheckpointRestorer.Result result = TrainerBestModelCheckpointRestorer.restore(
                true,
                checkpoint,
                tempDir.resolve("manifest.properties"),
                true,
                (name, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> {
                    throw loadFailure;
                });

        assertFalse(result.restored());
        assertEquals("cannot decode", result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertTrue(result.failure().getMessage().contains("Failed to restore best model checkpoint"));
        assertSame(loadFailure, result.failure().getCause());
    }
}
