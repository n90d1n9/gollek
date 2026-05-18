package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Applies manifest integrity checks and strict/lenient resume failure policy.
 */
final class TrainerCheckpointIntegrityGate {
    private TrainerCheckpointIntegrityGate() {
    }

    static Decision evaluate(
            String artifactName,
            Path checkpointFile,
            Path manifestFile,
            String failureMessagePrefix,
            boolean failOnMismatch,
            IntegrityChecker integrityChecker) {
        TrainerCheckpointCompatibilityReport report = integrityChecker.check(artifactName, checkpointFile);
        if (report.compatible()) {
            return Decision.compatibleResult();
        }
        RuntimeException failure = failOnMismatch
                ? new IllegalStateException(failureMessagePrefix + " at " + manifestFile + ": " + report.error())
                : null;
        return Decision.mismatchResult(report.error(), failure);
    }

    @FunctionalInterface
    interface IntegrityChecker {
        TrainerCheckpointCompatibilityReport check(String artifactName, Path checkpointFile);
    }

    record Decision(boolean compatible, String loadError, RuntimeException failure) {
        static Decision compatibleResult() {
            return new Decision(true, null, null);
        }

        static Decision mismatchResult(String loadError, RuntimeException failure) {
            return new Decision(false, loadError, failure);
        }
    }
}
