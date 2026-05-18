package tech.kayys.gollek.ml.train;

import java.nio.file.Path;

/**
 * Persists model checkpoints and their compatibility metadata.
 */
final class TrainerModelCheckpointWriter {
    private TrainerModelCheckpointWriter() {
    }

    static WriteResult writeModel(Path checkpointFile, TrainerCheckpointIO.CheckedPathWriter modelWriter) {
        if (checkpointFile == null) {
            return WriteResult.skipped();
        }
        try {
            TrainerCheckpointIO.writeAtomically(checkpointFile, modelWriter);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    static WriteResult writeMetadata(
            Path metadataFile,
            Path checkpointFile,
            TrainerModelCheckpointMetadata.ExpectedModel expectedModel,
            int metadataVersion) {
        if (metadataFile == null) {
            return WriteResult.skipped();
        }
        try {
            TrainerModelCheckpointMetadata.write(metadataFile, checkpointFile, expectedModel, metadataVersion);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    record WriteResult(boolean written, String error) {
        static WriteResult skipped() {
            return new WriteResult(false, null);
        }

        static WriteResult success() {
            return new WriteResult(true, null);
        }

        static WriteResult failed(Exception error) {
            return new WriteResult(false, error.getMessage());
        }
    }
}
