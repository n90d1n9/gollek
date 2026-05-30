package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TrainerRuntimeArtifactStateTest {
    @Test
    void recordsRuntimeArtifactLoadAndSaveState() {
        TrainerRuntimeArtifactState state = new TrainerRuntimeArtifactState();
        state.recordHistoryMissingOnResume();
        state.recordHistoryLoadError("history mismatch");
        state.recordHistoryPersistence(TrainerRuntimeArtifactPersistence.SaveResult.failed("history save failed"));
        state.recordReportPersistence(TrainerRuntimeArtifactPersistence.SaveResult.savedResult());
        state.recordManifestPersistence(TrainerRuntimeArtifactPersistence.SaveResult.failed("manifest save failed"));
        state.recordRuntimeResumeDecision(TrainerRuntimeCheckpointResume.Decision.skippedForMismatch(
                "runtime mismatch"));

        TrainerSummaryAssembler.RuntimeArtifacts artifacts = state.runtimeArtifacts(
                Path.of("history.csv"),
                Path.of("report.json"),
                Path.of("manifest.json"),
                Path.of("runtime.json"));

        assertTrue(state.historyMissingOnResume());
        assertTrue(artifacts.history().missingOnResume());
        assertFalse(artifacts.history().loaded());
        assertEquals("history mismatch", artifacts.history().loadError());
        assertEquals("history save failed", artifacts.history().saveError());
        assertTrue(artifacts.report().saved());
        assertEquals("manifest save failed", artifacts.manifest().saveError());
        assertFalse(artifacts.runtimeCheckpoint().resumeAllowed());
        assertTrue(artifacts.runtimeCheckpoint().integrityMismatch());
        assertTrue(artifacts.runtimeCheckpoint().resumeSkipped());
        assertEquals("runtime mismatch", artifacts.runtimeCheckpoint().loadError());
        assertEquals(
                "runtime-checkpoint-integrity-mismatch",
                artifacts.runtimeCheckpoint().decisionCode());
        assertEquals(
                "skip runtime checkpoint and rebuild runtime state from trainer artifacts",
                artifacts.runtimeCheckpoint().recommendedAction());
    }

    @Test
    void successfulHistoryLoadClearsMissingAndLoadError() {
        TrainerRuntimeArtifactState state = new TrainerRuntimeArtifactState();
        state.recordHistoryMissingOnResume();
        state.recordHistoryLoadError("stale");

        state.recordHistoryLoaded();

        TrainerSummaryAssembler.RuntimeArtifacts artifacts = state.runtimeArtifacts(
                Path.of("history.csv"),
                null,
                null,
                null);
        assertFalse(state.historyMissingOnResume());
        assertTrue(artifacts.history().loaded());
        assertFalse(artifacts.history().missingOnResume());
        assertNull(artifacts.history().loadError());
    }

    @Test
    void manifestIntegrityCheckUpdatesManifestStateAndDiagnostics() {
        TrainerRuntimeArtifactState state = new TrainerRuntimeArtifactState();
        TrainerCheckpointResumeDiagnostics diagnostics = new TrainerCheckpointResumeDiagnostics();

        state.recordManifestIntegrityCheck(
                new TrainerCheckpointArtifactIntegrity.Result(
                        "runtime",
                        TrainerCheckpointCompatibilityReport.incompatible("checksum mismatch"),
                        true,
                        false,
                        null,
                        true,
                        false),
                diagnostics);

        TrainerSummaryAssembler.RuntimeArtifacts artifacts = state.runtimeArtifacts(
                null,
                null,
                Path.of("manifest.json"),
                Path.of("runtime.json"));

        assertTrue(artifacts.manifest().loaded());
        assertFalse(artifacts.manifest().missingOnResume());
        assertTrue(artifacts.manifestIntegrityMismatch());
        assertEquals("runtime: checksum mismatch", diagnostics.compatibilityMismatches().getFirst());
    }
}
