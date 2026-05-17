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
        if (gamma <= 0) {
            throw new IllegalArgumentException("gamma must be positive, got: " + gamma);
        }
        this.initialLr = optimizer.learningRate();
        this.stepSize = stepSize;
        this.gamma = gamma;
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
        requireFloatMatch(state.get("initialLr"), initialLr, "initialLr");
        requireIntMatch(state.get("stepSize"), stepSize, "stepSize");
        requireFloatMatch(state.get("gamma"), gamma, "gamma");
        this.step = Math.max(0, readInt(state.get("step"), step));
        setLearningRate(readFloat(state.get("currentLr"), computeLearningRateForStep(this.step)));
    }

    private float computeLearningRateForStep(int targetStep) {
        int decayCount = Math.max(0, targetStep) / stepSize;
        float newLr = initialLr;
        for (int i = 0; i < decayCount; i++) {
            newLr *= gamma;
        }
        return newLr;
    }

    private static void requireIntMatch(Object value, int expected, String fieldName) {
        if (value == null) {
            return;
        }
        int loaded = readInt(value, expected);
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid StepLR checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static void requireFloatMatch(Object value, float expected, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected);
        if (Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid StepLR checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static int readInt(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static float readFloat(Object value, float fallback) {
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof String text) {
            try {
                return Float.parseFloat(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    @Override
    public String toString() {
        return "StepLR(stepSize=" + stepSize + ", gamma=" + gamma + ")";
    }
}
