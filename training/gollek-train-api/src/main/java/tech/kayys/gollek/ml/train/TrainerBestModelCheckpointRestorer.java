package tech.kayys.gollek.ml.train;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Restores the selected best-model checkpoint while keeping restore policy out of the trainer loop.
 */
final class TrainerBestModelCheckpointRestorer {
    private TrainerBestModelCheckpointRestorer() {
    }

    static Result restore(
            boolean restoreRequested,
            Path checkpointFile,
            Path manifestFile,
            boolean failOnCheckpointLoadError,
            IntegrityChecker integrityChecker,
            ModelLoader modelLoader) {
        if (!restoreRequested || checkpointFile == null || !Files.isRegularFile(checkpointFile)) {
            return Result.skipped();
        }
        TrainerCheckpointCompatibilityReport integrity = integrityChecker.check("bestModel", checkpointFile);
        if (!integrity.compatible()) {
            RuntimeException failure = failOnCheckpointLoadError
                    ? new IllegalStateException(
                            "Best model checkpoint integrity mismatch before restore at "
                                    + manifestFile + ": " + integrity.error())
                    : null;
            return Result.failed(integrity.error(), failure);
        }
        TrainerModelCheckpointLoader.LoadResult result =
                TrainerModelCheckpointLoader.loadModel(checkpointFile, modelLoader::load);
        if (result.loaded()) {
            return Result.restoredResult();
        }
        if (result.error() == null) {
            return Result.skipped();
        }
        RuntimeException failure = failOnCheckpointLoadError
                ? new IllegalStateException(
                        "Failed to restore best model checkpoint from " + checkpointFile,
                        result.cause())
                : null;
        return Result.failed(result.error(), failure);
    }

    @FunctionalInterface
    interface IntegrityChecker {
        TrainerCheckpointCompatibilityReport check(String artifactName, Path artifactFile);
    }

    @FunctionalInterface
    interface ModelLoader {
        void load(Path checkpointFile) throws Exception;
    }

    record Result(boolean restored, String loadError, RuntimeException failure) {
        static Result skipped() {
            return new Result(false, null, null);
        }

        static Result restoredResult() {
            return new Result(true, null, null);
        }

        static Result failed(String loadError, RuntimeException failure) {
            return new Result(false, loadError, failure);
        }
    }
}
