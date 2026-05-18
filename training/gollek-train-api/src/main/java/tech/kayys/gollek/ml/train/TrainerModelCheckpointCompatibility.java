package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Adapts model checkpoint metadata checks into trainer-facing resume state.
 */
final class TrainerModelCheckpointCompatibility {
    private TrainerModelCheckpointCompatibility() {
    }

    static Result check(
            Path metadataFile,
            Path checkpointFile,
            TrainerModelCheckpointMetadata.ExpectedModel expectedModel,
            int supportedMetadataVersion) {
        TrainerModelCheckpointMetadata.CompatibilityCheck check = TrainerModelCheckpointMetadata.check(
                metadataFile,
                checkpointFile,
                expectedModel,
                supportedMetadataVersion);
        return new Result(
                check.report(),
                check.loaded(),
                check.missing(),
                check.loadError(),
                !check.report().compatible());
    }

    record Result(
            TrainerCheckpointCompatibilityReport report,
            boolean metadataLoaded,
            boolean metadataMissing,
            String metadataLoadError,
            boolean compatibilityMismatch) {
        void recordMismatch(TrainerCheckpointResumeDiagnostics diagnostics) {
            if (compatibilityMismatch && diagnostics != null) {
                diagnostics.recordCompatibilityMismatch("model", report.error());
            }
        }
    }
}
