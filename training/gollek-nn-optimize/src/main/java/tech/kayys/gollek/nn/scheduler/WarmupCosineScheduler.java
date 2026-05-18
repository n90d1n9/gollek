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
        float validatedMaxLr = SchedulerValidation.positive(maxLr, "maxLr");
        float validatedMinLr = SchedulerValidation.nonNegative(minLr, "minLr");
        if (validatedMinLr > validatedMaxLr) {
            throw new IllegalArgumentException(
                    "minLr must be in [0, maxLr], got: " + validatedMinLr
                            + " with maxLr=" + validatedMaxLr);
        }
        this.warmupSteps = warmupSteps;
        this.totalSteps  = totalSteps;
        this.maxLr       = validatedMaxLr;
        this.minLr       = validatedMinLr;
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
        SchedulerValidation.requireIntMatch(
                state.get("warmupSteps"), warmupSteps, "WarmupCosineScheduler", "warmupSteps");
        SchedulerValidation.requireIntMatch(
                state.get("totalSteps"), totalSteps, "WarmupCosineScheduler", "totalSteps");
        SchedulerValidation.requireFloatMatch(
                state.get("maxLr"), maxLr, "WarmupCosineScheduler", "maxLr");
        SchedulerValidation.requireFloatMatch(
                state.get("minLr"), minLr, "WarmupCosineScheduler", "minLr");
        currentStep = SchedulerValidation.readNonNegativeInt(
                state.get("currentStep"), currentStep, "WarmupCosineScheduler", "currentStep");
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"), computeLearningRateForStep(currentStep),
                "WarmupCosineScheduler", "currentLr");
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

}
