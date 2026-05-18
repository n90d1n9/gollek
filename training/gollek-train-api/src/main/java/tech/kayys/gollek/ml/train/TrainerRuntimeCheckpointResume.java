package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Decides whether the underlying runtime may resume from its checkpoint.
 */
final class TrainerRuntimeCheckpointResume {
    private TrainerRuntimeCheckpointResume() {
    }

    static Decision evaluate(
            boolean resumeFromCheckpoint,
            Path runtimeCheckpointFile,
            IntegrityChecker integrityChecker) {
        if (!resumeFromCheckpoint || runtimeCheckpointFile == null) {
            return Decision.allowed(resumeFromCheckpoint);
        }
        TrainerCheckpointCompatibilityReport integrity =
                integrityChecker.check("runtime", runtimeCheckpointFile);
        if (integrity.compatible()) {
            return Decision.allowed(true);
        }
        return Decision.skippedForMismatch(integrity.error());
    }

    @FunctionalInterface
    interface IntegrityChecker {
        TrainerCheckpointCompatibilityReport check(String artifactName, Path artifactFile);
    }

    record Decision(
            boolean resumeAllowed,
            boolean integrityMismatch,
            boolean resumeSkipped,
            String loadError) {
        static Decision allowed(boolean resumeAllowed) {
            return new Decision(resumeAllowed, false, false, null);
        }

        static Decision skippedForMismatch(String error) {
            return new Decision(false, true, true, error);
        }

        boolean shouldFail(boolean failOnCheckpointLoadError) {
            return failOnCheckpointLoadError && integrityMismatch;
        }
    }
}
