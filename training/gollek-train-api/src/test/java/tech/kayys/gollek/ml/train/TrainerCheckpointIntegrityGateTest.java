package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class TrainerCheckpointIntegrityGateTest {

    @Test
    void compatibleArtifactPassesThroughWithoutError() {
        AtomicReference<String> artifact = new AtomicReference<>();
        Path checkpoint = Path.of("optimizer.state");

        TrainerCheckpointIntegrityGate.Decision decision = TrainerCheckpointIntegrityGate.evaluate(
                "optimizer",
                checkpoint,
                Path.of("manifest.properties"),
                "Optimizer checkpoint integrity mismatch for resume",
                true,
                (name, path) -> {
                    artifact.set(name + ":" + path);
                    return TrainerCheckpointCompatibilityReport.ok();
                });

        assertTrue(decision.compatible());
        assertNull(decision.loadError());
        assertNull(decision.failure());
        assertEquals("optimizer:optimizer.state", artifact.get());
    }

    @Test
    void mismatchCanBeRecordedWithoutThrowingInLenientMode() {
        TrainerCheckpointIntegrityGate.Decision decision = TrainerCheckpointIntegrityGate.evaluate(
                "scheduler",
                Path.of("scheduler.state"),
                Path.of("manifest.properties"),
                "Scheduler checkpoint integrity mismatch for resume",
                false,
                (name, path) -> TrainerCheckpointCompatibilityReport.incompatible("scheduler checksum mismatch"));

        assertFalse(decision.compatible());
        assertEquals("scheduler checksum mismatch", decision.loadError());
        assertNull(decision.failure());
    }

    @Test
    void mismatchCreatesCanonicalFailureInStrictMode() {
        Path manifest = Path.of("checkpoint-manifest.properties");

        TrainerCheckpointIntegrityGate.Decision decision = TrainerCheckpointIntegrityGate.evaluate(
                "gradScaler",
                Path.of("grad-scaler.state"),
                manifest,
                "GradScaler checkpoint integrity mismatch for resume",
                true,
                (name, path) -> TrainerCheckpointCompatibilityReport.incompatible("gradScaler size mismatch"));

        assertFalse(decision.compatible());
        assertEquals("gradScaler size mismatch", decision.loadError());
        assertInstanceOf(IllegalStateException.class, decision.failure());
        assertEquals(
                "GradScaler checkpoint integrity mismatch for resume at "
                        + manifest + ": gradScaler size mismatch",
                decision.failure().getMessage());
    }
}
