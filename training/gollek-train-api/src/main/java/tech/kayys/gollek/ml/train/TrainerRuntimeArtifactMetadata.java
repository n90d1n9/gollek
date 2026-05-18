package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Publishes trainer runtime artifact state for history, reports, manifests, and runtime checkpoints.
 */
final class TrainerRuntimeArtifactMetadata {
    private TrainerRuntimeArtifactMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            Artifact history,
            SaveOnlyArtifact report,
            Artifact manifest,
            boolean manifestIntegrityMismatch,
            RuntimeCheckpoint runtimeCheckpoint) {
        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "trainingHistory",
                history.enabled(),
                history.file(),
                history.missingOnResume(),
                history.loaded(),
                history.saved(),
                history.loadError(),
                history.saveError());
        TrainerSummaryMetadata.putSaveOnlyArtifactStatus(
                metadata,
                "trainingReport",
                report.enabled(),
                report.file(),
                report.saved(),
                report.saveError());
        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "checkpointManifest",
                manifest.enabled(),
                manifest.file(),
                manifest.missingOnResume(),
                manifest.loaded(),
                manifest.saved(),
                manifest.loadError(),
                manifest.saveError());
        metadata.put("checkpointManifestIntegrityMismatch", manifestIntegrityMismatch);
        metadata.put("runtimeCheckpointPresent", TrainerMetadataSupport.filePresent(runtimeCheckpoint.file()));
        metadata.put("runtimeCheckpointIntegrityMismatch", runtimeCheckpoint.integrityMismatch());
        metadata.put("runtimeCheckpointResumeSkipped", runtimeCheckpoint.resumeSkipped());
        metadata.put("runtimeCheckpointLoadFailed", runtimeCheckpoint.loadError() != null);
    }

    record Artifact(
            boolean enabled,
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
    }

    record SaveOnlyArtifact(
            boolean enabled,
            Path file,
            boolean saved,
            String saveError) {
    }

    record RuntimeCheckpoint(
            Path file,
            boolean integrityMismatch,
            boolean resumeSkipped,
            String loadError) {
    }
}
