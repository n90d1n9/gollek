package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerCheckpointLoadGateTest {
    @TempDir
    Path tempDir;

    @Test
    void skipsWhenResumeDisabledOrCheckpointIsNotConfigured() {
        AtomicBoolean attempted = new AtomicBoolean(false);

        TrainerCheckpointLoadGate.Decision disabled = TrainerCheckpointLoadGate.evaluate(
                false,
                tempDir.resolve("model.safetensors"),
                attempted,
                "model",
                true);
        TrainerCheckpointLoadGate.Decision unconfigured = TrainerCheckpointLoadGate.evaluate(
                true,
                null,
                attempted,
                "model",
                true);

        assertFalse(disabled.shouldLoad());
        assertFalse(unconfigured.shouldLoad());
        assertFalse(disabled.missing());
        assertFalse(unconfigured.missing());
        assertFalse(attempted.get());
    }

    @Test
    void loadsExistingArtifactOnlyOncePerAttemptFlag() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("optimizer.state"), "state");
        AtomicBoolean attempted = new AtomicBoolean(false);

        TrainerCheckpointLoadGate.Decision first = TrainerCheckpointLoadGate.evaluate(
                true,
                checkpoint,
                attempted,
                "optimizer",
                true);
        TrainerCheckpointLoadGate.Decision second = TrainerCheckpointLoadGate.evaluate(
                true,
                checkpoint,
                attempted,
                "optimizer",
                true);

        assertTrue(first.shouldLoad());
        assertFalse(first.missing());
        assertNull(first.failure());
        assertFalse(second.shouldLoad());
        assertFalse(second.missing());
        assertTrue(attempted.get());
    }

    @Test
    void missingArtifactCanFailFastWithCanonicalMessage() {
        Path checkpoint = tempDir.resolve("missing.state");
        AtomicBoolean attempted = new AtomicBoolean(false);

        TrainerCheckpointLoadGate.Decision decision = TrainerCheckpointLoadGate.evaluate(
                true,
                checkpoint,
                attempted,
                "scheduler",
                true);

        assertFalse(decision.shouldLoad());
        assertTrue(decision.missing());
        assertInstanceOf(IllegalStateException.class, decision.failure());
        assertTrue(decision.failure().getMessage()
                .contains("Missing scheduler checkpoint artifact for resume: " + checkpoint));
        assertTrue(attempted.get());
    }

    @Test
    void missingArtifactCanBeRecordedWithoutFailure() {
        AtomicBoolean attempted = new AtomicBoolean(false);

        TrainerCheckpointLoadGate.Decision decision = TrainerCheckpointLoadGate.evaluate(
                true,
                tempDir.resolve("history.csv"),
                attempted,
                "history",
                false);

        assertFalse(decision.shouldLoad());
        assertTrue(decision.missing());
        assertNull(decision.failure());
        assertTrue(attempted.get());
    }
}
