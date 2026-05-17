package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Linear warmup followed by cosine annealing learning rate schedule.
 *
 * <p>Used widely in Transformer fine-tuning (BERT, GPT, LLaMA).
 * During the warmup phase the learning rate increases linearly from 0 to
 * {@code maxLr}; after warmup it follows a cosine decay down to {@code minLr}.
 *
 * <pre>
 *   step &lt; warmupSteps:  lr = maxLr * step / warmupSteps
 *   step &ge; warmupSteps: lr = minLr + 0.5*(maxLr-minLr)*(1 + cos(π*(step-warmup)/(total-warmup)))
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var scheduler = new WarmupCosineScheduler(optimizer,
 *     warmupSteps = 100, totalSteps = 1000,
 *     maxLr = 3e-4f, minLr = 1e-6f);
 *
 * for (int step = 0; step < 1000; step++) {
 *     optimizer.step();
 *     scheduler.step();
 * }
 * }</pre>
 */
public final class WarmupCosineScheduler extends LRScheduler {

    private final int   warmupSteps;
    private final int   totalSteps;
    private final float maxLr;
    private final float minLr;
    private int   currentStep = 0;
    private float currentLr;

    /**
     * Constructs a warmup-cosine scheduler.
     *
     * @param optimizer    the optimizer whose learning rate will be updated
     * @param warmupSteps  number of linear warmup steps
     * @param totalSteps   total training steps (warmup + cosine decay)
     * @param maxLr        peak learning rate (reached at end of warmup)
     * @param minLr        minimum learning rate (reached at end of cosine decay)
     */
    public WarmupCosineScheduler(Optimizer optimizer, int warmupSteps, int totalSteps,
                                  float maxLr, float minLr) {
        super(optimizer);
        if (warmupSteps < 0) {
            throw new IllegalArgumentException("warmupSteps must be non-negative, got: " + warmupSteps);
        }
        if (totalSteps <= 0) {
            throw new IllegalArgumentException("totalSteps must be positive, got: " + totalSteps);
        }
        if (warmupSteps > totalSteps) {
            throw new IllegalArgumentException(
                    "warmupSteps must be <= totalSteps, got: " + warmupSteps + " > " + totalSteps);
        }
        if (maxLr <= 0) {
            throw new IllegalArgumentException("maxLr must be positive, got: " + maxLr);
        }
        if (minLr < 0 || minLr > maxLr) {
            throw new IllegalArgumentException(
                    "minLr must be in [0, maxLr], got: " + minLr + " with maxLr=" + maxLr);
        }
        this.warmupSteps = warmupSteps;
        this.totalSteps  = totalSteps;
        this.maxLr       = maxLr;
        this.minLr       = minLr;
        this.currentLr   = 0f; // starts at 0
        setLearningRate(currentLr);
    }

    /**
     * Advances the scheduler by one step and updates the optimizer's learning rate.
     *
     * <p>Call this <em>after</em> {@code optimizer.step()} each training step.
     */
    @Override
    public void step() {
        currentStep++;
        currentLr = computeLearningRateForStep(currentStep);
        setLearningRate(currentLr);
    }

    /**
     * Returns the current learning rate.
     *
     * @return current lr value
     */
    @Override
    public float getLr() { return currentLr; }

    /** @return current step count */
    public int currentStep() { return currentStep; }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "WarmupCosineScheduler");
        state.put("warmupSteps", warmupSteps);
        state.put("totalSteps", totalSteps);
        state.put("maxLr", maxLr);
        state.put("minLr", minLr);
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
        if (schedulerName instanceof String name && !"WarmupCosineScheduler".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected WarmupCosineScheduler but got " + name);
        }
        requireIntMatch(state.get("warmupSteps"), warmupSteps, "warmupSteps");
        requireIntMatch(state.get("totalSteps"), totalSteps, "totalSteps");
        requireFloatMatch(state.get("maxLr"), maxLr, "maxLr");
        requireFloatMatch(state.get("minLr"), minLr, "minLr");
        currentStep = Math.max(0, readInt(state.get("currentStep"), currentStep));
        currentLr = readFloat(state.get("currentLr"), computeLearningRateForStep(currentStep));
        setLearningRate(currentLr);
    }

    private float computeLearningRateForStep(int targetStep) {
        int clampedStep = Math.max(0, Math.min(targetStep, totalSteps));
        if (warmupSteps > 0 && clampedStep <= warmupSteps) {
            return maxLr * (float) clampedStep / warmupSteps;
        }
        if (totalSteps == warmupSteps) {
            return minLr;
        }
        float progress = (float) (clampedStep - warmupSteps) / (totalSteps - warmupSteps);
        return minLr + 0.5f * (maxLr - minLr)
                * (1f + (float) Math.cos(Math.PI * progress));
    }

    private static void requireIntMatch(Object value, int expected, String fieldName) {
        if (value == null) {
            return;
        }
        int loaded = readInt(value, expected);
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid WarmupCosineScheduler checkpoint payload: " + fieldName
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
                    "Invalid WarmupCosineScheduler checkpoint payload: " + fieldName
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
}
