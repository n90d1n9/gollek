package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Adapts manifest artifact checks into trainer-facing resume state.
 */
final class TrainerCheckpointArtifactIntegrity {
    private TrainerCheckpointArtifactIntegrity() {
    }

    static Result check(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        TrainerCheckpointManifest.CompatibilityCheck check = TrainerCheckpointManifest.checkArtifact(
                manifestFile,
                artifactName,
                artifactFile,
                supportedManifestVersion);
        return result(artifactName, check);
    }

    static Result checkRequired(
            Path manifestFile,
            String artifactName,
            Path artifactFile,
            int supportedManifestVersion) {
        TrainerCheckpointManifest.CompatibilityCheck check = TrainerCheckpointManifest.checkRequiredArtifact(
                manifestFile,
                artifactName,
                artifactFile,
                supportedManifestVersion);
        return result(artifactName, check);
    }

    private static Result result(String artifactName, TrainerCheckpointManifest.CompatibilityCheck check) {
        return new Result(
                artifactName,
                check.report(),
                check.loaded(),
                check.missing(),
                check.loadError(),
                !check.report().compatible(),
                check.manifestEntryMissing());
    }

    record Result(
            String artifactName,
            TrainerCheckpointCompatibilityReport report,
            boolean manifestLoaded,
            boolean manifestMissing,
            String manifestLoadError,
            boolean integrityMismatch,
            boolean manifestEntryMissing) {
        void recordMismatch(TrainerCheckpointResumeDiagnostics diagnostics) {
            if (integrityMismatch && diagnostics != null) {
                if (manifestEntryMissing) {
                    diagnostics.recordManifestEntryMissing(artifactName);
                }
                diagnostics.recordCompatibilityMismatch(artifactName, report.error());
            }
        }
    }
}
