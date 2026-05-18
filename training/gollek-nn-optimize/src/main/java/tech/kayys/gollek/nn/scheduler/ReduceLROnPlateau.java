package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Reduces the optimizer learning rate when a validation loss or metric stops
 * improving.
 *
 * <p>This scheduler is meant to be stepped after validation with
 * {@link #step(double)}. The first finite metric initializes the best value.
 * Later non-improving values increment the bad-step counter; once that counter
 * exceeds {@code patience}, the current learning rate is multiplied by
 * {@code factor} and clamped to {@code minLr}.</p>
 */
public final class ReduceLROnPlateau extends LRScheduler {

    public enum Mode {
        MIN {
            @Override
            boolean isImproved(double current, double best, double threshold) {
                return current < best - threshold;
            }
        },
        MAX {
            @Override
            boolean isImproved(double current, double best, double threshold) {
                return current > best + threshold;
            }
        };

        abstract boolean isImproved(double current, double best, double threshold);
    }

    private final Mode mode;
    private final float factor;
    private final int patience;
    private final double threshold;
    private final int cooldown;
    private final float minLr;

    private int stepCount;
    private int badSteps;
    private int cooldownRemaining;
    private int reductionCount;
    private double bestMetric = Double.NaN;
    private float currentLr;

    public ReduceLROnPlateau(
            Optimizer optimizer,
            Mode mode,
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        super(optimizer);
        if (patience < 0) {
            throw new IllegalArgumentException("patience must be non-negative, got: " + patience);
        }
        if (!Double.isFinite(threshold) || threshold < 0.0) {
            throw new IllegalArgumentException("threshold must be finite and non-negative, got: " + threshold);
        }
        if (cooldown < 0) {
            throw new IllegalArgumentException("cooldown must be non-negative, got: " + cooldown);
        }
        this.mode = mode == null ? Mode.MIN : mode;
        this.factor = SchedulerValidation.factor(factor, "factor");
        this.patience = patience;
        this.threshold = threshold;
        this.cooldown = cooldown;
        this.minLr = SchedulerValidation.nonNegative(minLr, "minLr");
        this.currentLr = SchedulerValidation.learningRate(optimizer.learningRate(), "currentLr");
    }

    @Override
    public void step() {
        step(Double.NaN);
    }

    @Override
    public void step(double metric) {
        stepCount++;
        if (!Double.isFinite(metric)) {
            return;
        }

        if (Double.isNaN(bestMetric) || mode.isImproved(metric, bestMetric, threshold)) {
            bestMetric = metric;
            badSteps = 0;
            return;
        }

        if (cooldownRemaining > 0) {
            cooldownRemaining--;
            badSteps = 0;
            return;
        }

        badSteps++;
        if (badSteps > patience) {
            reduceLearningRate();
            cooldownRemaining = cooldown;
            badSteps = 0;
        }
    }

    @Override
    public float getLr() {
        return currentLr;
    }

    public int stepCount() {
        return stepCount;
    }

    public int badSteps() {
        return badSteps;
    }

    public int reductionCount() {
        return reductionCount;
    }

    public double bestMetric() {
        return bestMetric;
    }

    private void reduceLearningRate() {
        float nextLr = Math.max(minLr, currentLr * factor);
        if (nextLr < currentLr - 1.0e-12f) {
            currentLr = nextLr;
            setLearningRate(currentLr);
            reductionCount++;
        } else {
            currentLr = optimizer.learningRate();
        }
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "ReduceLROnPlateau");
        state.put("mode", mode.name());
        state.put("factor", factor);
        state.put("patience", patience);
        state.put("threshold", threshold);
        state.put("cooldown", cooldown);
        state.put("minLr", minLr);
        state.put("stepCount", stepCount);
        state.put("badSteps", badSteps);
        state.put("cooldownRemaining", cooldownRemaining);
        state.put("reductionCount", reductionCount);
        state.put("bestMetric", bestMetric);
        state.put("currentLr", currentLr);
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"ReduceLROnPlateau".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected ReduceLROnPlateau but got " + name);
        }
        requireModeMatch(state.get("mode"));
        SchedulerValidation.requireFloatMatch(state.get("factor"), factor, "ReduceLROnPlateau", "factor");
        SchedulerValidation.requireIntMatch(state.get("patience"), patience, "ReduceLROnPlateau", "patience");
        SchedulerValidation.requireDoubleMatch(
                state.get("threshold"), threshold, "ReduceLROnPlateau", "threshold");
        SchedulerValidation.requireIntMatch(state.get("cooldown"), cooldown, "ReduceLROnPlateau", "cooldown");
        SchedulerValidation.requireFloatMatch(state.get("minLr"), minLr, "ReduceLROnPlateau", "minLr");
        stepCount = SchedulerValidation.readNonNegativeInt(
                state.get("stepCount"), stepCount, "ReduceLROnPlateau", "stepCount");
        badSteps = SchedulerValidation.readNonNegativeInt(
                state.get("badSteps"), badSteps, "ReduceLROnPlateau", "badSteps");
        cooldownRemaining = SchedulerValidation.readNonNegativeInt(
                state.get("cooldownRemaining"), cooldownRemaining,
                "ReduceLROnPlateau", "cooldownRemaining");
        reductionCount = SchedulerValidation.readNonNegativeInt(
                state.get("reductionCount"), reductionCount, "ReduceLROnPlateau", "reductionCount");
        bestMetric = SchedulerValidation.readFiniteOrNaNDouble(
                state.get("bestMetric"), bestMetric, "ReduceLROnPlateau", "bestMetric");
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"), currentLr, "ReduceLROnPlateau", "currentLr");
        setLearningRate(currentLr);
    }

    private void requireModeMatch(Object value) {
        if (value == null) {
            return;
        }
        Mode loaded = Mode.valueOf(String.valueOf(value));
        if (loaded != mode) {
            throw new IllegalArgumentException(
                    "Invalid ReduceLROnPlateau checkpoint payload: mode mismatch (expected "
                            + mode + ", got " + loaded + ")");
        }
    }

}
