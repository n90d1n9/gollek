package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Coordinates model checkpoint resume without leaking decision logic into the trainer.
 */
final class TrainerModelCheckpointResume {
    private TrainerModelCheckpointResume() {
    }

    static Result resume(
            boolean resumeRequested,
            Path checkpointFile,
            Path metadataFile,
            AtomicBoolean loadAttempted,
            TrainerModelCheckpointMetadata.ExpectedModel expectedModel,
            int metadataVersion,
            boolean failOnLoadError,
            TrainerCheckpointResumeDiagnostics diagnostics,
            TrainerModelCheckpointLoader.CheckedPathLoader modelLoader) {
        TrainerCheckpointLoadGate.Decision gate = TrainerCheckpointLoadGate.evaluate(
                resumeRequested,
                checkpointFile,
                loadAttempted,
                "model",
                failOnLoadError);
        if (gate.failure() != null) {
            return Result.failure(true, gate.failure());
        }
        if (gate.missing()) {
            return Result.missing();
        }
        if (!gate.shouldLoad()) {
            return Result.skipped();
        }

        TrainerModelCheckpointCompatibility.Result compatibility = TrainerModelCheckpointCompatibility.check(
                metadataFile,
                checkpointFile,
                expectedModel,
                metadataVersion);
        if (compatibility.compatibilityMismatch()) {
            compatibility.recordMismatch(diagnostics);
            RuntimeException failure = failOnLoadError
                    ? new IllegalStateException(
                            "Model checkpoint metadata mismatch for resume at "
                                    + metadataFile + ": " + compatibility.report().error())
                    : null;
            return Result.compatibilityMismatch(
                    compatibility.metadataLoaded(),
                    compatibility.metadataMissing(),
                    compatibility.metadataLoadError(),
                    compatibility.report().error(),
                    failure);
        }

        TrainerModelCheckpointLoader.LoadResult load = TrainerModelCheckpointLoader.loadModel(
                checkpointFile,
                modelLoader);
        if (load.loaded()) {
            return Result.loaded(
                    compatibility.metadataLoaded(),
                    compatibility.metadataMissing(),
                    compatibility.metadataLoadError());
        }
        RuntimeException failure = failOnLoadError && load.error() != null
                ? new IllegalStateException("Failed to load model checkpoint from " + checkpointFile, load.cause())
                : null;
        return Result.loadFailed(
                compatibility.metadataLoaded(),
                compatibility.metadataMissing(),
                compatibility.metadataLoadError(),
                load.error(),
                failure);
    }

    record Result(
            boolean stateChanged,
            boolean missingOnResume,
            boolean loaded,
            boolean compatibilityMismatch,
            boolean metadataLoaded,
            boolean metadataMissingOnResume,
            String metadataLoadError,
            String loadError,
            RuntimeException failure) {
        static Result skipped() {
            return new Result(false, false, false, false, false, false, null, null, null);
        }

        static Result missing() {
            return new Result(true, true, false, false, false, false, null, null, null);
        }

        static Result failure(boolean missingOnResume, RuntimeException failure) {
            return new Result(true, missingOnResume, false, false, false, false, null, null, failure);
        }

        static Result compatibilityMismatch(
                boolean metadataLoaded,
                boolean metadataMissingOnResume,
                String metadataLoadError,
                String loadError,
                RuntimeException failure) {
            return new Result(
                    true,
                    false,
                    false,
                    true,
                    metadataLoaded,
                    metadataMissingOnResume,
                    metadataLoadError,
                    loadError,
                    failure);
        }

        static Result loaded(
                boolean metadataLoaded,
                boolean metadataMissingOnResume,
                String metadataLoadError) {
            return new Result(
                    true,
                    false,
                    true,
                    false,
                    metadataLoaded,
                    metadataMissingOnResume,
                    metadataLoadError,
                    null,
                    null);
        }

        static Result loadFailed(
                boolean metadataLoaded,
                boolean metadataMissingOnResume,
                String metadataLoadError,
                String loadError,
                RuntimeException failure) {
            return new Result(
                    true,
                    false,
                    false,
                    false,
                    metadataLoaded,
                    metadataMissingOnResume,
                    metadataLoadError,
                    loadError,
                    failure);
        }
    }
}
