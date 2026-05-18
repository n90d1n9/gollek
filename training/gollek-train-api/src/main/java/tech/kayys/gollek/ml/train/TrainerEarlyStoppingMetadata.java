package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Publishes early-stopping configuration and monitor state for trainer summaries.
 */
final class TrainerEarlyStoppingMetadata {
    private TrainerEarlyStoppingMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            int patience,
            double minDelta,
            String monitorLabel,
            CanonicalTrainer.BestModelMonitorMode monitorMode,
            boolean metricDriven,
            MonitorState monitorState,
            boolean triggered,
            int stopEpoch) {
        metadata.put("earlyStoppingEnabled", patience > 0);
        metadata.put("earlyStoppingPatience", Math.max(0, patience));
        metadata.put("earlyStoppingMinDelta", Math.max(0.0, minDelta));
        metadata.put("earlyStoppingMonitor", monitorLabel);
        metadata.put("earlyStoppingMonitorMode", monitorMode.name());
        metadata.put("earlyStoppingMonitorMetricDriven", metricDriven);
        metadata.put("earlyStoppingMonitorBestEpoch", monitorState.bestEpoch());
        metadata.put("earlyStoppingMonitorBestValue", monitorState.bestValue());
        metadata.put("earlyStoppingMonitorLatestValue", monitorState.latestValue());
        metadata.put("earlyStoppingMonitorEpochsWithoutImprovement", monitorState.epochsWithoutImprovement());
        if (triggered) {
            metadata.put("earlyStoppingTriggered", true);
            metadata.put("earlyStoppingEpoch", stopEpoch);
            metadata.put("stopReason", "early-stopping");
        }
    }

    record MonitorState(
            int bestEpoch,
            double bestValue,
            double latestValue,
            int epochsWithoutImprovement) {
    }
}
