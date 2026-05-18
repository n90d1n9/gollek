package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Tracks best-model checkpoint monitor state independently from checkpoint IO.
 */
final class TrainerBestModelCheckpointMonitor {
    private final boolean enabled;
    private final String metricName;
    private final CanonicalTrainer.BestModelMonitorMode mode;
    private final double minDelta;
    private volatile boolean saved;
    private volatile boolean restored;
    private volatile int epoch = -1;
    private volatile double validationLoss = Double.NaN;
    private volatile double monitorValue = Double.NaN;

    TrainerBestModelCheckpointMonitor(
            boolean enabled,
            String metricName,
            CanonicalTrainer.BestModelMonitorMode mode,
            double minDelta) {
        this.enabled = enabled;
        this.metricName = metricName;
        this.mode = mode == null ? CanonicalTrainer.BestModelMonitorMode.MIN : mode;
        this.minDelta = Math.max(0.0, minDelta);
    }

    Decision evaluate(int epoch, double validationLoss, Map<String, Double> validationMetrics) {
        if (!enabled || !Double.isFinite(validationLoss)) {
            return Decision.skip();
        }
        double currentMonitorValue = TrainerMonitorSupport.valueOrThrow(
                metricName,
                validationLoss,
                validationMetrics,
                "Best model monitor metric");
        if (!Double.isFinite(currentMonitorValue)) {
            return Decision.skip();
        }
        boolean improved = Double.isNaN(monitorValue)
                || mode.isImproved(currentMonitorValue, monitorValue, minDelta);
        if (!improved) {
            return Decision.skip();
        }
        return Decision.save(epoch, validationLoss, currentMonitorValue);
    }

    void markSaved(Decision decision) {
        if (!decision.shouldSave()) {
            return;
        }
        saved = true;
        epoch = decision.epoch();
        validationLoss = decision.validationLoss();
        monitorValue = decision.monitorValue();
    }

    void markRestored() {
        restored = true;
    }

    TrainerBestModelCheckpointMetadata.State state() {
        return new TrainerBestModelCheckpointMetadata.State(
                saved,
                restored,
                epoch,
                validationLoss,
                monitorValue);
    }

    record Decision(
            boolean shouldSave,
            int epoch,
            double validationLoss,
            double monitorValue) {
        static Decision skip() {
            return new Decision(false, -1, Double.NaN, Double.NaN);
        }

        static Decision save(int epoch, double validationLoss, double monitorValue) {
            return new Decision(true, epoch, validationLoss, monitorValue);
        }
    }
}
