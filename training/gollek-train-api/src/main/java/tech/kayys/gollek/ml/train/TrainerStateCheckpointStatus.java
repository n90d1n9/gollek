package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns optimizer, scheduler, and GradScaler checkpoint lifecycle status.
 */
final class TrainerStateCheckpointStatus {
    private final Checkpoint optimizer = new Checkpoint();
    private final Checkpoint scheduler = new Checkpoint();
    private final Checkpoint gradScaler = new Checkpoint();
    private volatile boolean gradScalerFallbackUsed;

    AtomicBoolean optimizerLoadAttempted() {
        return optimizer.loadAttempted;
    }

    AtomicBoolean schedulerLoadAttempted() {
        return scheduler.loadAttempted;
    }

    AtomicBoolean gradScalerLoadAttempted() {
        return gradScaler.loadAttempted;
    }

    void recordOptimizerResume(TrainerStateCheckpointResume.Result result) {
        optimizer.recordResume(result);
    }

    void recordSchedulerResume(TrainerStateCheckpointResume.Result result) {
        scheduler.recordResume(result);
    }

    void recordGradScalerResume(TrainerStateCheckpointResume.Result result) {
        gradScaler.recordResume(result);
        if (!result.stateChanged()) {
            return;
        }
        if (result.loaded()) {
            gradScalerFallbackUsed = false;
        } else if (!result.missingOnResume() && result.loadError() != null) {
            gradScalerFallbackUsed = true;
        }
    }

    void recordOptimizerPersistence(TrainerStateCheckpointPersistence.Result result) {
        optimizer.recordPersistence(result);
    }

    void recordSchedulerPersistence(TrainerStateCheckpointPersistence.Result result) {
        scheduler.recordPersistence(result);
    }

    void recordGradScalerPersistence(TrainerStateCheckpointPersistence.Result result) {
        gradScaler.recordPersistence(result);
    }

    TrainerSummaryAssembler.StateCheckpoint optimizerSummary(Path checkpointFile, boolean supported) {
        return optimizer.summary(checkpointFile, supported);
    }

    TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint schedulerSummary(
            Path checkpointFile,
            boolean supported,
            boolean resumeRequested) {
        return new TrainerLearningRateSchedulerMetadata.SchedulerCheckpoint(
                checkpointFile != null,
                supported,
                resumeRequested,
                checkpointFile,
                scheduler.missingOnResume,
                scheduler.loaded,
                scheduler.saved,
                scheduler.loadError,
                scheduler.saveError);
    }

    TrainerMixedPrecisionMetadata.GradScalerCheckpoint gradScalerSummary(
            Path checkpointFile,
            boolean enabled,
            boolean supported,
            boolean resumeRequested) {
        return new TrainerMixedPrecisionMetadata.GradScalerCheckpoint(
                enabled,
                supported,
                resumeRequested,
                checkpointFile,
                gradScaler.missingOnResume,
                gradScaler.loaded,
                gradScaler.saved,
                gradScaler.loadError,
                gradScaler.saveError,
                gradScalerFallbackUsed);
    }

    boolean optimizerMissingOnResume() {
        return optimizer.missingOnResume;
    }

    boolean schedulerMissingOnResume() {
        return scheduler.missingOnResume;
    }

    boolean gradScalerMissingOnResume() {
        return gradScaler.missingOnResume;
    }

    String optimizerLoadError() {
        return optimizer.loadError;
    }

    String optimizerSaveError() {
        return optimizer.saveError;
    }

    String schedulerLoadError() {
        return scheduler.loadError;
    }

    String schedulerSaveError() {
        return scheduler.saveError;
    }

    String gradScalerLoadError() {
        return gradScaler.loadError;
    }

    String gradScalerSaveError() {
        return gradScaler.saveError;
    }

    private static final class Checkpoint {
        private final AtomicBoolean loadAttempted = new AtomicBoolean(false);
        private volatile boolean missingOnResume;
        private volatile boolean loaded;
        private volatile boolean saved;
        private volatile String loadError;
        private volatile String saveError;

        private void recordResume(TrainerStateCheckpointResume.Result result) {
            if (!result.stateChanged()) {
                return;
            }
            missingOnResume = result.missingOnResume();
            loaded = result.loaded();
            loadError = result.loadError();
        }

        private void recordPersistence(TrainerStateCheckpointPersistence.Result result) {
            if (result.saved()) {
                saved = true;
                saveError = null;
            } else if (result.saveError() != null) {
                saveError = result.saveError();
            }
        }

        private TrainerSummaryAssembler.StateCheckpoint summary(Path checkpointFile, boolean supported) {
            return new TrainerSummaryAssembler.StateCheckpoint(
                    checkpointFile != null,
                    supported,
                    checkpointFile,
                    missingOnResume,
                    loaded,
                    saved,
                    loadError,
                    saveError);
        }
    }
}
