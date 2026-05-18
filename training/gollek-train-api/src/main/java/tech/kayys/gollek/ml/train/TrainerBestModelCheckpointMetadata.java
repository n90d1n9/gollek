package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Publishes best-model checkpoint state for trainer summaries.
 */
final class TrainerBestModelCheckpointMetadata {
    private TrainerBestModelCheckpointMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            boolean enabled,
            boolean restoreRequested,
            Path checkpointFile,
            String monitorLabel,
            CanonicalTrainer.BestModelMonitorMode monitorMode,
            State state,
            String saveError,
            String loadError) {
        metadata.put("bestModelCheckpointEnabled", enabled);
        metadata.put("bestModelCheckpointRestoreRequested", restoreRequested);
        metadata.put("bestModelCheckpointSaved", state.saved());
        metadata.put("bestModelCheckpointRestored", state.restored());
        metadata.put("bestModelCheckpointPresent", TrainerMetadataSupport.filePresent(checkpointFile));
        metadata.put("bestModelCheckpointMonitor", monitorLabel);
        metadata.put("bestModelCheckpointMonitorMode", monitorMode.name());
        metadata.put("bestModelCheckpointEpoch", state.epoch());
        metadata.put("bestModelCheckpointValidationLoss", state.validationLoss());
        metadata.put("bestModelCheckpointMonitorValue", state.monitorValue());
        metadata.put("bestModelCheckpointSaveFailed", saveError != null);
        metadata.put("bestModelCheckpointLoadFailed", loadError != null);
    }

    record State(
            boolean saved,
            boolean restored,
            int epoch,
            double validationLoss,
            double monitorValue) {
    }
}
