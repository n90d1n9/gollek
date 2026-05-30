package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns latest-model checkpoint and metadata lifecycle status.
 */
final class TrainerModelCheckpointStatus {
    private final AtomicBoolean loadAttempted = new AtomicBoolean(false);
    private volatile boolean missingOnResume;
    private volatile boolean loaded;
    private volatile boolean saved;
    private volatile boolean metadataLoaded;
    private volatile boolean metadataSaved;
    private volatile boolean metadataMissingOnResume;
    private volatile boolean compatibilityMismatch;
    private volatile String loadError;
    private volatile String saveError;
    private volatile String metadataLoadError;
    private volatile String metadataSaveError;

    AtomicBoolean loadAttempted() {
        return loadAttempted;
    }

    void recordResume(TrainerModelCheckpointResume.Result result) {
        if (!result.stateChanged()) {
            return;
        }
        missingOnResume = result.missingOnResume();
        loaded = result.loaded();
        compatibilityMismatch = result.compatibilityMismatch();
        metadataLoaded = result.metadataLoaded();
        metadataMissingOnResume = result.metadataMissingOnResume();
        metadataLoadError = result.metadataLoadError();
        loadError = result.loadError();
    }

    void recordModelPersistence(TrainerModelCheckpointWriter.WriteResult result) {
        if (result.written()) {
            saved = true;
            saveError = null;
        } else if (result.error() != null) {
            saveError = result.error();
        }
    }

    void recordMetadataPersistence(TrainerModelCheckpointWriter.WriteResult result) {
        if (result.written()) {
            metadataSaved = true;
            metadataSaveError = null;
        } else if (result.error() != null) {
            metadataSaveError = result.error();
        }
    }

    TrainerSummaryAssembler.ModelCheckpoint modelSummary(Path checkpointFile) {
        return new TrainerSummaryAssembler.ModelCheckpoint(
                checkpointFile,
                missingOnResume,
                loaded,
                saved,
                loadError,
                saveError,
                compatibilityMismatch);
    }

    TrainerSummaryAssembler.Artifact metadataSummary(Path metadataFile) {
        return new TrainerSummaryAssembler.Artifact(
                metadataFile,
                metadataMissingOnResume,
                metadataLoaded,
                metadataSaved,
                metadataLoadError,
                metadataSaveError);
    }

    boolean missingOnResume() {
        return missingOnResume;
    }

    String loadError() {
        return loadError;
    }

    String saveError() {
        return saveError;
    }

    String metadataLoadError() {
        return metadataLoadError;
    }

    String metadataSaveError() {
        return metadataSaveError;
    }
}
