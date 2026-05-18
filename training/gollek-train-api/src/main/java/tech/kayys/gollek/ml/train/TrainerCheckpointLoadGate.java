package tech.kayys.gollek.ml.train;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Decides whether a checkpoint artifact should be loaded during resume.
 */
final class TrainerCheckpointLoadGate {
    private TrainerCheckpointLoadGate() {
    }

    static Decision evaluate(
            boolean resumeRequested,
            Path checkpointFile,
            AtomicBoolean loadAttempted,
            String artifactName,
            boolean failOnMissing) {
        if (!resumeRequested || checkpointFile == null) {
            return Decision.skippedResult();
        }
        if (loadAttempted != null && !loadAttempted.compareAndSet(false, true)) {
            return Decision.skippedResult();
        }
        if (!Files.isRegularFile(checkpointFile)) {
            RuntimeException failure = failOnMissing
                    ? TrainerCheckpointResumeDiagnostics.missingArtifactException(artifactName, checkpointFile)
                    : null;
            return Decision.missingResult(failure);
        }
        return Decision.loadResult();
    }

    record Decision(boolean shouldLoad, boolean missing, RuntimeException failure) {
        static Decision skippedResult() {
            return new Decision(false, false, null);
        }

        static Decision missingResult(RuntimeException failure) {
            return new Decision(false, true, failure);
        }

        static Decision loadResult() {
            return new Decision(true, false, null);
        }
    }
}
