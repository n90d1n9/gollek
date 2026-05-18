package tech.kayys.gollek.ml.train;

import java.util.Map;
import tech.kayys.gollek.ml.optim.LRScheduler;

/**
 * Owns scheduler step routing and resumeable step-count telemetry.
 */
final class TrainerLearningRateSchedulerStepper {
    private final LRScheduler scheduler;
    private final CanonicalTrainer.SchedulerStepUnit stepUnit;
    private int stepCount;

    TrainerLearningRateSchedulerStepper(
            LRScheduler scheduler,
            CanonicalTrainer.SchedulerStepUnit stepUnit) {
        this.scheduler = scheduler;
        this.stepUnit = stepUnit == null ? CanonicalTrainer.SchedulerStepUnit.BATCH : stepUnit;
    }

    boolean step(CanonicalTrainer.SchedulerStepUnit unit) {
        return step(unit, Double.NaN);
    }

    boolean step(CanonicalTrainer.SchedulerStepUnit unit, double monitorValue) {
        if (scheduler == null || stepUnit != unit) {
            return false;
        }
        if (unit == CanonicalTrainer.SchedulerStepUnit.VALIDATION) {
            scheduler.step(monitorValue);
        } else {
            scheduler.step();
        }
        stepCount++;
        return true;
    }

    boolean enabled() {
        return scheduler != null;
    }

    boolean supportsStateDict() {
        return scheduler != null && scheduler.supportsStateDict();
    }

    String schedulerType() {
        return scheduler == null ? "none" : scheduler.getClass().getSimpleName();
    }

    Map<String, Object> stateSnapshot() {
        if (!supportsStateDict()) {
            return Map.of();
        }
        return TrainerMetadataSupport.stateSnapshot(scheduler.stateDict());
    }

    int stepCount() {
        return stepCount;
    }

    void restoreStepCount(int stepCount) {
        this.stepCount = Math.max(0, stepCount);
    }
}
