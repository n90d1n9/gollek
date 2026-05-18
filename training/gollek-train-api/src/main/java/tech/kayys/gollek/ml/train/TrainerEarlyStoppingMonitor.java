package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Tracks metric-driven early stopping independently from the trainer loop.
 */
final class TrainerEarlyStoppingMonitor {
    private final String metricName;
    private final int patience;
    private final double minDelta;
    private final CanonicalTrainer.BestModelMonitorMode mode;
    private volatile int bestEpoch = -1;
    private volatile int epochsWithoutImprovement;
    private volatile int stopEpoch = -1;
    private volatile double bestValue = Double.NaN;
    private volatile double latestValue = Double.NaN;
    private volatile boolean triggered;

    TrainerEarlyStoppingMonitor(
            String metricName,
            int patience,
            double minDelta,
            CanonicalTrainer.BestModelMonitorMode mode) {
        this.metricName = metricName;
        this.patience = Math.max(0, patience);
        this.minDelta = Math.max(0.0, minDelta);
        this.mode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MIN : mode;
    }

    UpdateResult update(int epoch, double validationLoss, Map<String, Double> validationMetrics) {
        if (metricName == null || patience <= 0 || triggered) {
            return UpdateResult.keepTraining();
        }
        double monitorValue = TrainerMonitorSupport.valueOrThrow(
                metricName,
                validationLoss,
                validationMetrics,
                "Early stopping monitor metric");
        latestValue = monitorValue;
        if (!Double.isFinite(monitorValue)) {
            return UpdateResult.keepTraining();
        }
        if (Double.isNaN(bestValue) || mode.isImproved(monitorValue, bestValue, minDelta)) {
            bestValue = monitorValue;
            bestEpoch = epoch;
            epochsWithoutImprovement = 0;
            return UpdateResult.improvedResult();
        }
        epochsWithoutImprovement++;
        if (epochsWithoutImprovement >= patience) {
            triggered = true;
            stopEpoch = epoch;
            return UpdateResult.stop();
        }
        return UpdateResult.keepTraining();
    }

    TrainerEarlyStoppingMetadata.MonitorState state() {
        return new TrainerEarlyStoppingMetadata.MonitorState(
                bestEpoch,
                bestValue,
                latestValue,
                epochsWithoutImprovement);
    }

    boolean triggered() {
        return triggered;
    }

    int stopEpoch() {
        return stopEpoch;
    }

    record UpdateResult(boolean improved, boolean shouldStop) {
        static UpdateResult improvedResult() {
            return new UpdateResult(true, false);
        }

        static UpdateResult keepTraining() {
            return new UpdateResult(false, false);
        }

        static UpdateResult stop() {
            return new UpdateResult(false, true);
        }
    }
}
