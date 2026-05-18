package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Publishes learning-rate scheduler and scheduler checkpoint state for trainer summaries.
 */
final class TrainerLearningRateSchedulerMetadata {
    private TrainerLearningRateSchedulerMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            boolean enabled,
            String stepUnit,
            int stepCount,
            String schedulerType,
            String monitorLabel,
            boolean metricDriven,
            Map<String, Object> schedulerState,
            double learningRate,
            SchedulerCheckpoint checkpoint) {
        Map<String, Object> state = TrainerMetadataSupport.stateSnapshot(schedulerState);
        metadata.put("learningRateSchedulerEnabled", enabled);
        metadata.put("learningRateSchedulerStepUnit", stepUnit);
        metadata.put("learningRateSchedulerStepCount", stepCount);
        metadata.put("learningRateSchedulerType", schedulerType);
        metadata.put("learningRateSchedulerMonitor", monitorLabel);
        metadata.put("learningRateSchedulerMonitorMetricDriven", metricDriven);
        metadata.put("learningRateSchedulerState", state);
        TrainerMetadataSupport.flatten(metadata, "learningRateSchedulerState.", state);
        metadata.put("learningRate", learningRate);
        TrainerSummaryMetadata.putSupportedCheckpointStatus(
                metadata,
                "schedulerCheckpoint",
                checkpoint.enabled(),
                checkpoint.supported(),
                checkpoint.resumeRequested(),
                checkpoint.file(),
                checkpoint.missingOnResume(),
                checkpoint.loaded(),
                checkpoint.saved(),
                checkpoint.loadError(),
                checkpoint.saveError());
    }

    record SchedulerCheckpoint(
            boolean enabled,
            boolean supported,
            boolean resumeRequested,
            Path file,
            boolean missingOnResume,
            boolean loaded,
            boolean saved,
            String loadError,
            String saveError) {
    }
}
