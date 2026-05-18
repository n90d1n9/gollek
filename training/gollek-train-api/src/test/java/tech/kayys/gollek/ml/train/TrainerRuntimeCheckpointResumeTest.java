package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class TrainerRuntimeCheckpointResumeTest {

    @Test
    void disabledResumeDoesNotCheckRuntimeArtifact() {
        AtomicInteger checks = new AtomicInteger();

        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                false,
                Path.of("runtime.state"),
                (artifact, file) -> {
                    checks.incrementAndGet();
                    return TrainerCheckpointCompatibilityReport.ok();
                });

        assertFalse(decision.resumeAllowed());
        assertFalse(decision.integrityMismatch());
        assertFalse(decision.resumeSkipped());
        assertNull(decision.loadError());
        assertEquals(0, checks.get());
    }

    @Test
    void missingRuntimeCheckpointConfigurationAllowsRuntimeResumeWithoutCheck() {
        AtomicInteger checks = new AtomicInteger();

        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                true,
                null,
                (artifact, file) -> {
                    checks.incrementAndGet();
                    return TrainerCheckpointCompatibilityReport.incompatible("should not run");
                });

        assertTrue(decision.resumeAllowed());
        assertFalse(decision.integrityMismatch());
        assertFalse(decision.resumeSkipped());
        assertNull(decision.loadError());
        assertEquals(0, checks.get());
    }

    @Test
    void compatibleRuntimeArtifactAllowsResume() {
        AtomicReference<String> artifactName = new AtomicReference<>();
        AtomicReference<Path> artifactFile = new AtomicReference<>();

        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                true,
                Path.of("runtime.state"),
                (artifact, file) -> {
                    artifactName.set(artifact);
                    artifactFile.set(file);
                    return TrainerCheckpointCompatibilityReport.ok();
                });

        assertTrue(decision.resumeAllowed());
        assertFalse(decision.integrityMismatch());
        assertFalse(decision.resumeSkipped());
        assertNull(decision.loadError());
        assertEquals("runtime", artifactName.get());
        assertEquals(Path.of("runtime.state"), artifactFile.get());
    }

    @Test
    void incompatibleRuntimeArtifactSkipsResumeAndCanFailFast() {
        TrainerRuntimeCheckpointResume.Decision decision = TrainerRuntimeCheckpointResume.evaluate(
                true,
                Path.of("runtime.state"),
                (artifact, file) -> TrainerCheckpointCompatibilityReport.incompatible("sha mismatch"));

        assertFalse(decision.resumeAllowed());
        assertTrue(decision.integrityMismatch());
        assertTrue(decision.resumeSkipped());
        assertEquals("sha mismatch", decision.loadError());
        assertFalse(decision.shouldFail(false));
        assertTrue(decision.shouldFail(true));
    }
}
