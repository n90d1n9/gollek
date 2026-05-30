package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Multiplicative exponential learning-rate scheduler.
 *
 * <p>Each call to {@link #step()} updates the learning rate to
 * {@code initialLr * gamma ^ step}. Values of {@code gamma < 1} decay the
 * learning rate, while {@code gamma > 1} can be used for controlled growth
 * schedules.</p>
 */
public final class ExponentialLR extends LRScheduler {
    private final float initialLr;
    private final float gamma;
    private int currentStep;
    private float currentLr;

    public ExponentialLR(Optimizer optimizer, float gamma) {
        super(optimizer);
        this.initialLr = SchedulerValidation.learningRate(optimizer.learningRate(), "initialLr");
        this.gamma = SchedulerValidation.positive(gamma, "gamma");
        this.currentLr = initialLr;
    }

    @Override
    public void step() {
        currentStep++;
        currentLr = computeLearningRateForStep(currentStep);
        setLearningRate(currentLr);
    }

    @Override
    public float getLr() {
        return currentLr;
    }

    public int currentStep() {
        return currentStep;
    }

    public float initialLr() {
        return initialLr;
    }

    public float gamma() {
        return gamma;
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "ExponentialLR");
        state.put("initialLr", initialLr);
        state.put("gamma", gamma);
        state.put("currentStep", currentStep);
        state.put("currentLr", currentLr);
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"ExponentialLR".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected ExponentialLR but got " + name);
        }
        SchedulerValidation.requireFloatMatch(state.get("initialLr"), initialLr, "ExponentialLR", "initialLr");
        SchedulerValidation.requireFloatMatch(state.get("gamma"), gamma, "ExponentialLR", "gamma");
        currentStep = SchedulerValidation.readNonNegativeInt(
                state.get("currentStep"), currentStep, "ExponentialLR", "currentStep");
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"), computeLearningRateForStep(currentStep), "ExponentialLR", "currentLr");
        setLearningRate(currentLr);
    }

    private float computeLearningRateForStep(int targetStep) {
        return (float) (initialLr * Math.pow(gamma, Math.max(0, targetStep)));
    }

    @Override
    public String toString() {
        return "ExponentialLR(gamma=" + gamma + ")";
    }
}
