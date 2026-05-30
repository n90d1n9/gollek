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
            return resumeFromCheckpoint ? Decision.notConfigured() : Decision.resumeNotRequested();
        }
        TrainerCheckpointCompatibilityReport integrity =
                integrityChecker.check("runtime", runtimeCheckpointFile);
        if (integrity.compatible()) {
            return Decision.compatible();
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
            String loadError,
            String decisionCode,
            String recommendedAction) {
        static Decision allowed(boolean resumeAllowed) {
            return resumeAllowed ? compatible() : resumeNotRequested();
        }

        static Decision resumeNotRequested() {
            return new Decision(
                    false,
                    false,
                    false,
                    null,
                    "runtime-resume-not-requested",
                    "run without runtime checkpoint resume");
        }

        static Decision notConfigured() {
            return new Decision(
                    true,
                    false,
                    false,
                    null,
                    "runtime-checkpoint-not-configured",
                    "continue without runtime checkpoint integrity check");
        }

        static Decision compatible() {
            return new Decision(
                    true,
                    false,
                    false,
                    null,
                    "runtime-checkpoint-compatible",
                    "continue with runtime checkpoint resume");
        }

        static Decision skippedForMismatch(String error) {
            return new Decision(
                    false,
                    true,
                    true,
                    error,
                    "runtime-checkpoint-integrity-mismatch",
                    "skip runtime checkpoint and rebuild runtime state from trainer artifacts");
        }

        boolean shouldFail(boolean failOnCheckpointLoadError) {
            return failOnCheckpointLoadError && integrityMismatch;
        }
    }
}
