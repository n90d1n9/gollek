package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Shared resume flow for state-dictionary checkpoints such as optimizer, scheduler, and GradScaler.
 */
final class TrainerStateCheckpointResume {
    private TrainerStateCheckpointResume() {
    }

    static Result resume(
            boolean resumeRequested,
            Path checkpointFile,
            AtomicBoolean loadAttempted,
            String missingArtifactName,
            String manifestArtifactName,
            Path manifestFile,
            String integrityFailurePrefix,
            String loadFailureArtifactName,
            boolean failOnLoadError,
            TrainerCheckpointIntegrityGate.IntegrityChecker integrityChecker,
            StateReader reader,
            StateValidator validator,
            StateApplier applier) {
        TrainerCheckpointLoadGate.Decision gate = TrainerCheckpointLoadGate.evaluate(
                resumeRequested,
                checkpointFile,
                loadAttempted,
                missingArtifactName,
                failOnLoadError);
        if (gate.missing()) {
            return Result.missing(gate.failure());
        }
        if (gate.failure() != null) {
            return Result.failed(null, gate.failure());
        }
        if (!gate.shouldLoad()) {
            return Result.skipped();
        }

        TrainerCheckpointIntegrityGate.Decision integrity = TrainerCheckpointIntegrityGate.evaluate(
                manifestArtifactName,
                checkpointFile,
                manifestFile,
                integrityFailurePrefix,
                failOnLoadError,
                integrityChecker);
        if (!integrity.compatible()) {
            return Result.failed(integrity.loadError(), integrity.failure());
        }

        try {
            StateSnapshot snapshot = reader.read(checkpointFile);
            validator.validate(snapshot.state());
            applier.apply(snapshot.state());
            return Result.loaded(snapshot.counter());
        } catch (Exception error) {
            RuntimeException failure = failOnLoadError
                    ? new IllegalStateException(
                            "Failed to load " + loadFailureArtifactName + " checkpoint from " + checkpointFile,
                            error)
                    : null;
            return Result.failed(error.getMessage(), failure);
        }
    }

    @FunctionalInterface
    interface StateReader {
        StateSnapshot read(Path checkpointFile) throws Exception;
    }

    @FunctionalInterface
    interface StateValidator {
        void validate(Map<String, Object> state) throws Exception;
    }

    @FunctionalInterface
    interface StateApplier {
        void apply(Map<String, Object> state) throws Exception;
    }

    record StateSnapshot(Map<String, Object> state, int counter) {
    }

    record Result(
            boolean stateChanged,
            boolean missingOnResume,
            boolean loaded,
            String loadError,
            RuntimeException failure,
            int counter) {
        static Result skipped() {
            return new Result(false, false, false, null, null, 0);
        }

        static Result missing(RuntimeException failure) {
            return new Result(true, true, false, null, failure, 0);
        }

        static Result loaded(int counter) {
            return new Result(true, false, true, null, null, counter);
        }

        static Result failed(String loadError, RuntimeException failure) {
            return new Result(true, false, false, loadError, failure, 0);
        }
    }
}
