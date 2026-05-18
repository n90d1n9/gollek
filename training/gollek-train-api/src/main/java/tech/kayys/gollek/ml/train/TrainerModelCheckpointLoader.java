package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Loads model checkpoints while preserving the root load failure.
 */
final class TrainerModelCheckpointLoader {
    private TrainerModelCheckpointLoader() {
    }

    static LoadResult loadModel(Path checkpointFile, CheckedPathLoader modelLoader) {
        if (checkpointFile == null) {
            return LoadResult.skipped();
        }
        try {
            modelLoader.load(checkpointFile);
            return LoadResult.success();
        } catch (Exception error) {
            return LoadResult.failed(error);
        }
    }

    @FunctionalInterface
    interface CheckedPathLoader {
        void load(Path checkpointFile) throws Exception;
    }

    record LoadResult(boolean loaded, String error, Exception cause) {
        static LoadResult skipped() {
            return new LoadResult(false, null, null);
        }

        static LoadResult success() {
            return new LoadResult(true, null, null);
        }

        static LoadResult failed(Exception error) {
            return new LoadResult(false, error.getMessage(), error);
        }
    }
}
