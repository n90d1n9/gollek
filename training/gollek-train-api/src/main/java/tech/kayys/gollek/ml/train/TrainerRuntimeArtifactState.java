package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns runtime artifact load/save state published into training summaries.
 */
final class TrainerRuntimeArtifactState {
    private final AtomicBoolean historyLoadAttempted = new AtomicBoolean(false);
    private volatile boolean historyMissingOnResume;
    private volatile boolean historyLoaded;
    private volatile boolean historySaved;
    private volatile boolean reportSaved;
    private volatile boolean manifestLoaded;
    private volatile boolean manifestSaved;
    private volatile boolean manifestMissingOnResume;
    private volatile boolean manifestIntegrityMismatch;
    private volatile boolean runtimeCheckpointResumeAllowed;
    private volatile boolean runtimeCheckpointIntegrityMismatch;
    private volatile boolean runtimeCheckpointResumeSkipped;
    private volatile String historyLoadError;
    private volatile String historySaveError;
    private volatile String reportSaveError;
    private volatile String manifestLoadError;
    private volatile String manifestSaveError;
    private volatile String runtimeCheckpointLoadError;
    private volatile String runtimeCheckpointDecisionCode = "runtime-checkpoint-unchecked";
    private volatile String runtimeCheckpointRecommendedAction = "runtime checkpoint resume decision was not evaluated";

    AtomicBoolean historyLoadAttempted() {
        return historyLoadAttempted;
    }

    void recordRuntimeResumeDecision(TrainerRuntimeCheckpointResume.Decision decision) {
        runtimeCheckpointResumeAllowed = decision.resumeAllowed();
        runtimeCheckpointIntegrityMismatch = decision.integrityMismatch();
        runtimeCheckpointResumeSkipped = decision.resumeSkipped();
        runtimeCheckpointLoadError = decision.loadError();
        runtimeCheckpointDecisionCode = decision.decisionCode();
        runtimeCheckpointRecommendedAction = decision.recommendedAction();
    }

    void recordManifestIntegrityCheck(
            TrainerCheckpointArtifactIntegrity.Result check,
            TrainerCheckpointResumeDiagnostics diagnostics) {
        manifestLoaded = check.manifestLoaded();
        manifestMissingOnResume = check.manifestMissing();
        manifestLoadError = check.manifestLoadError();
        if (check.integrityMismatch()) {
            manifestIntegrityMismatch = true;
            check.recordMismatch(diagnostics);
        }
    }

    void recordHistoryMissingOnResume() {
        historyMissingOnResume = true;
    }

    void recordHistoryLoaded() {
        historyLoaded = true;
        historyMissingOnResume = false;
        historyLoadError = null;
    }

    void recordHistoryLoadError(String error) {
        historyLoadError = error;
    }

    void recordHistoryPersistence(TrainerRuntimeArtifactPersistence.SaveResult result) {
        if (result.saved()) {
            historySaved = true;
            historySaveError = null;
        } else if (result.error() != null) {
            historySaveError = result.error();
        }
    }

    void recordReportPersistence(TrainerRuntimeArtifactPersistence.SaveResult result) {
        if (result.saved()) {
            reportSaved = true;
            reportSaveError = null;
        } else if (result.error() != null) {
            reportSaveError = result.error();
        }
    }

    void recordManifestPersistence(TrainerRuntimeArtifactPersistence.SaveResult result) {
        if (result.saved()) {
            manifestSaved = true;
            manifestSaveError = null;
        } else if (result.error() != null) {
            manifestSaveError = result.error();
        }
    }

    TrainerSummaryAssembler.RuntimeArtifacts runtimeArtifacts(
            Path historyFile,
            Path reportFile,
            Path manifestFile,
            Path runtimeCheckpointFile) {
        return new TrainerSummaryAssembler.RuntimeArtifacts(
                new TrainerRuntimeArtifactMetadata.Artifact(
                        historyFile != null,
                        historyFile,
                        historyMissingOnResume,
                        historyLoaded,
                        historySaved,
                        historyLoadError,
                        historySaveError),
                new TrainerRuntimeArtifactMetadata.SaveOnlyArtifact(
                        reportFile != null,
                        reportFile,
                        reportSaved,
                        reportSaveError),
                new TrainerRuntimeArtifactMetadata.Artifact(
                        manifestFile != null,
                        manifestFile,
                        manifestMissingOnResume,
                        manifestLoaded,
                        manifestSaved,
                        manifestLoadError,
                        manifestSaveError),
                manifestIntegrityMismatch,
                new TrainerRuntimeArtifactMetadata.RuntimeCheckpoint(
                        runtimeCheckpointFile,
                        runtimeCheckpointResumeAllowed,
                        runtimeCheckpointIntegrityMismatch,
                        runtimeCheckpointResumeSkipped,
                        runtimeCheckpointLoadError,
                        runtimeCheckpointDecisionCode,
                        runtimeCheckpointRecommendedAction));
    }

    boolean historyMissingOnResume() {
        return historyMissingOnResume;
    }

    String historyLoadError() {
        return historyLoadError;
    }

    String historySaveError() {
        return historySaveError;
    }

    String reportSaveError() {
        return reportSaveError;
    }

    String manifestLoadError() {
        return manifestLoadError;
    }

    String manifestSaveError() {
        return manifestSaveError;
    }

    String runtimeCheckpointLoadError() {
        return runtimeCheckpointLoadError;
    }
}
