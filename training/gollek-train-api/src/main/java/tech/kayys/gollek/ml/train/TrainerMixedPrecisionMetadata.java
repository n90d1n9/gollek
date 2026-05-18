package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Publishes mixed-precision and GradScaler checkpoint state for trainer summaries.
 */
final class TrainerMixedPrecisionMetadata {
    private TrainerMixedPrecisionMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            boolean enabled,
            double lossScale,
            boolean overflowDetected,
            int overflowSkipCount,
            Map<String, Object> gradScalerState,
            GradScalerCheckpoint checkpoint) {
        Map<String, Object> state = TrainerMetadataSupport.stateSnapshot(gradScalerState);
        metadata.put("mixedPrecisionEnabled", enabled);
        metadata.put("mixedPrecisionLossScale", lossScale);
        metadata.put("mixedPrecisionOverflowDetected", overflowDetected);
        metadata.put("mixedPrecisionOverflowSkipCount", overflowSkipCount);
        metadata.put("mixedPrecisionGradScalerState", state);
        TrainerMetadataSupport.flatten(metadata, "mixedPrecisionGradScalerState.", state);
        TrainerSummaryMetadata.putSupportedCheckpointStatus(
                metadata,
                "gradScalerCheckpoint",
                checkpoint.enabled(),
                checkpoint.supported(),
                checkpoint.resumeRequested(),
                checkpoint.file(),
                checkpoint.missingOnResume(),
                checkpoint.loaded(),
                checkpoint.saved(),
                checkpoint.loadError(),
                checkpoint.saveError());
        metadata.put("gradScalerCheckpointFallbackUsed", checkpoint.fallbackUsed());
    }

    record GradScalerCheckpoint(
            boolean enabled,
            boolean supported,
            boolean resumeRequested,
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError,
            boolean fallbackUsed) {
    }
}
