package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Step Decay Learning Rate Scheduler.
 * <p>
 * Multiplies learning rate by a factor (typically 0.1) every N steps/epochs.
 * This is one of the simplest and most effective learning rate schedules.
 * <p>
 * {@code lr = initial_lr * gamma ^ (step / step_size)}
 *
 * <h3>Example: Decay by 0.1 every 10 epochs</h3>
 * <pre>{@code
 * var scheduler = new StepLR(optimizer, 10, 0.1f);
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Training...
 *     scheduler.step();
 * }
 * // After epoch 10: lr *= 0.1
 * // After epoch 20: lr *= 0.1 (from original)
 * // etc.
 * }</pre>
 *
 * <h3>Typical Usage</h3>
 * Common configurations:
 * <ul>
 *   <li>StepLR(optimizer, 30, 0.1f) - decay by 0.1x every 30 epochs</li>
 *   <li>StepLR(optimizer, 10, 0.5f) - decay by 0.5x every 10 epochs</li>
 *   <li>StepLR(optimizer, 100, 0.1f) - decay by 0.1x every 100 epochs</li>
 * </ul>
 *
 * @see CosineAnnealingLR
 * @see ExponentialLR
 */
public class StepLR extends LRScheduler {

    private final float initialLr;
    private final int stepSize;
    private final float gamma;
    private int step = 0;

    /**
     * Create a StepLR scheduler.
     *
     * @param optimizer the optimizer to schedule
     * @param stepSize  number of steps before decay (e.g., epochs)
     * @param gamma     multiplicative factor (e.g., 0.1 for 10x decay)
     *
     * @throws IllegalArgumentException if stepSize <= 0 or gamma <= 0
     */
    public StepLR(Optimizer optimizer, int stepSize, float gamma) {
        super(optimizer);
        if (stepSize <= 0) {
            throw new IllegalArgumentException("stepSize must be positive, got: " + stepSize);
        }
        this.initialLr = SchedulerValidation.learningRate(optimizer.learningRate(), "initialLr");
        this.stepSize = stepSize;
        this.gamma = SchedulerValidation.positive(gamma, "gamma");
    }

    /**
     * Update learning rate for this step.
     */
    @Override
    public void step() {
        int decayCount = step / stepSize;
        float newLr = initialLr;
        for (int i = 0; i < decayCount; i++) {
            newLr *= gamma;
        }
        setLearningRate(newLr);
        step++;
    }

    @Override
    public float getLr() {
        return optimizer.learningRate();
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "StepLR");
        state.put("initialLr", initialLr);
        state.put("stepSize", stepSize);
        state.put("gamma", gamma);
        state.put("step", step);
        state.put("currentLr", optimizer.learningRate());
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"StepLR".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected StepLR but got " + name);
        }
        SchedulerValidation.requireFloatMatch(state.get("initialLr"), initialLr, "StepLR", "initialLr");
        SchedulerValidation.requireIntMatch(state.get("stepSize"), stepSize, "StepLR", "stepSize");
        SchedulerValidation.requireFloatMatch(state.get("gamma"), gamma, "StepLR", "gamma");
        this.step = SchedulerValidation.readNonNegativeInt(state.get("step"), step, "StepLR", "step");
        setLearningRate(SchedulerValidation.readLearningRate(
                state.get("currentLr"), computeLearningRateForStep(this.step), "StepLR", "currentLr"));
    }

    private float computeLearningRateForStep(int targetStep) {
        int decayCount = Math.max(0, targetStep) / stepSize;
        float newLr = initialLr;
        for (int i = 0; i < decayCount; i++) {
            newLr *= gamma;
        }
        return newLr;
    }

    @Override
    public String toString() {
        return "StepLR(stepSize=" + stepSize + ", gamma=" + gamma + ")";
    }
}
