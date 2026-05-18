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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerStateCheckpointResumeTest {
    @TempDir
    Path tempDir;

    @Test
    void compatibleStateCheckpointReadsValidatesAndAppliesState() throws Exception {
        Path checkpoint = writeCheckpoint("optimizer.state");
        AtomicReference<Map<String, Object>> validated = new AtomicReference<>();
        AtomicReference<Map<String, Object>> applied = new AtomicReference<>();

        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                checkpoint,
                new AtomicBoolean(false),
                "optimizer",
                "optimizer",
                tempDir.resolve("manifest.properties"),
                "Optimizer checkpoint integrity mismatch for resume",
                "optimizer",
                true,
                (artifact, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> new TrainerStateCheckpointResume.StateSnapshot(
                        new LinkedHashMap<>(Map.of("lr", 0.01)),
                        12),
                validated::set,
                applied::set);

        assertTrue(result.stateChanged());
        assertTrue(result.loaded());
        assertFalse(result.missingOnResume());
        assertNull(result.loadError());
        assertNull(result.failure());
        assertEquals(12, result.counter());
        assertEquals(0.01, validated.get().get("lr"));
        assertSame(validated.get(), applied.get());
    }

    @Test
    void integrityMismatchCanBeRecordedWithoutApplyingStateInLenientMode() throws Exception {
        Path checkpoint = writeCheckpoint("scheduler.state");
        AtomicBoolean readerCalled = new AtomicBoolean(false);
        AtomicBoolean applierCalled = new AtomicBoolean(false);

        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                checkpoint,
                new AtomicBoolean(false),
                "scheduler",
                "scheduler",
                tempDir.resolve("manifest.properties"),
                "Scheduler checkpoint integrity mismatch for resume",
                "scheduler",
                false,
                (artifact, path) -> TrainerCheckpointCompatibilityReport.incompatible("scheduler checksum mismatch"),
                path -> {
                    readerCalled.set(true);
                    return new TrainerStateCheckpointResume.StateSnapshot(Map.of(), 0);
                },
                state -> {
                },
                state -> applierCalled.set(true));

        assertTrue(result.stateChanged());
        assertFalse(result.loaded());
        assertFalse(result.missingOnResume());
        assertEquals("scheduler checksum mismatch", result.loadError());
        assertNull(result.failure());
        assertFalse(readerCalled.get());
        assertFalse(applierCalled.get());
    }

    @Test
    void missingStateCheckpointUsesCanonicalLoadGateFailure() {
        Path checkpoint = tempDir.resolve("missing.state");

        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                checkpoint,
                new AtomicBoolean(false),
                "GradScaler",
                "gradScaler",
                tempDir.resolve("manifest.properties"),
                "GradScaler checkpoint integrity mismatch for resume",
                "GradScaler",
                true,
                (artifact, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> new TrainerStateCheckpointResume.StateSnapshot(Map.of(), 0),
                state -> {
                },
                state -> {
                });

        assertTrue(result.stateChanged());
        assertTrue(result.missingOnResume());
        assertFalse(result.loaded());
        assertNull(result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertEquals(
                "Missing GradScaler checkpoint artifact for resume: " + checkpoint,
                result.failure().getMessage());
    }

    @Test
    void strictReadFailurePreservesCauseAndCanonicalMessage() throws Exception {
        Path checkpoint = writeCheckpoint("optimizer.state");
        IOException cause = new IOException("bad state");

        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                checkpoint,
                new AtomicBoolean(false),
                "optimizer",
                "optimizer",
                tempDir.resolve("manifest.properties"),
                "Optimizer checkpoint integrity mismatch for resume",
                "optimizer",
                true,
                (artifact, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> {
                    throw cause;
                },
                state -> {
                },
                state -> {
                });

        assertTrue(result.stateChanged());
        assertFalse(result.loaded());
        assertEquals("bad state", result.loadError());
        assertInstanceOf(IllegalStateException.class, result.failure());
        assertEquals(
                "Failed to load optimizer checkpoint from " + checkpoint,
                result.failure().getMessage());
        assertSame(cause, result.failure().getCause());
    }

    @Test
    void validatorFailurePreventsApplyingStateInLenientMode() throws Exception {
        Path checkpoint = writeCheckpoint("grad-scaler.state");
        AtomicBoolean applierCalled = new AtomicBoolean(false);

        TrainerStateCheckpointResume.Result result = TrainerStateCheckpointResume.resume(
                true,
                checkpoint,
                new AtomicBoolean(false),
                "GradScaler",
                "gradScaler",
                tempDir.resolve("manifest.properties"),
                "GradScaler checkpoint integrity mismatch for resume",
                "GradScaler",
                false,
                (artifact, path) -> TrainerCheckpointCompatibilityReport.ok(),
                path -> new TrainerStateCheckpointResume.StateSnapshot(Map.of("enabled", false), 3),
                state -> {
                    throw new IllegalArgumentException("mixed precision disabled");
                },
                state -> applierCalled.set(true));

        assertTrue(result.stateChanged());
        assertFalse(result.loaded());
        assertEquals("mixed precision disabled", result.loadError());
        assertNull(result.failure());
        assertFalse(applierCalled.get());
    }

    private Path writeCheckpoint(String name) throws IOException {
        return Files.writeString(tempDir.resolve(name), "state");
    }
}
