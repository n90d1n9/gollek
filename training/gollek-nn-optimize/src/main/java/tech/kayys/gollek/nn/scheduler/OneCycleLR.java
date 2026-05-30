package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * One-cycle learning rate schedule.
 *
 * <p>The schedule starts at {@code maxLr / divFactor}, warms up to
 * {@code maxLr} for the first {@code pctStart} fraction of training, then
 * anneals to {@code maxLr / divFactor / finalDivFactor}. This mirrors the
 * common PyTorch OneCycleLR workflow while staying optimizer-agnostic.</p>
 */
public final class OneCycleLR extends LRScheduler {

    public enum AnnealStrategy {
        COSINE,
        LINEAR
    }

    private final int totalSteps;
    private final int warmupSteps;
    private final float maxLr;
    private final float initialLr;
    private final float minLr;
    private final float pctStart;
    private final float divFactor;
    private final float finalDivFactor;
    private final AnnealStrategy annealStrategy;
    private int currentStep = 0;
    private float currentLr;

    public OneCycleLR(Optimizer optimizer, int totalSteps, float maxLr) {
        this(optimizer, totalSteps, maxLr, 0.3f, 25.0f, 10_000.0f, AnnealStrategy.COSINE);
    }

    public OneCycleLR(
            Optimizer optimizer,
            int totalSteps,
            float maxLr,
            float pctStart,
            float divFactor,
            float finalDivFactor,
            AnnealStrategy annealStrategy) {
        super(optimizer);
        if (totalSteps <= 0) {
            throw new IllegalArgumentException("totalSteps must be positive, got: " + totalSteps);
        }
        if (!Float.isFinite(pctStart) || pctStart <= 0.0f || pctStart >= 1.0f) {
            throw new IllegalArgumentException("pctStart must be finite and in (0, 1), got: " + pctStart);
        }
        this.maxLr = SchedulerValidation.positive(maxLr, "maxLr");
        this.divFactor = SchedulerValidation.positive(divFactor, "divFactor");
        this.finalDivFactor = SchedulerValidation.positive(finalDivFactor, "finalDivFactor");
        this.annealStrategy = annealStrategy == null ? AnnealStrategy.COSINE : annealStrategy;
        this.totalSteps = totalSteps;
        this.pctStart = pctStart;
        this.warmupSteps = Math.max(1, Math.round(totalSteps * pctStart));
        this.initialLr = this.maxLr / this.divFactor;
        this.minLr = this.initialLr / this.finalDivFactor;
        this.currentLr = initialLr;
        setLearningRate(currentLr);
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

    public int totalSteps() {
        return totalSteps;
    }

    public int warmupSteps() {
        return warmupSteps;
    }

    public float maxLr() {
        return maxLr;
    }

    public float initialLr() {
        return initialLr;
    }

    public float minLr() {
        return minLr;
    }

    public float pctStart() {
        return pctStart;
    }

    public AnnealStrategy annealStrategy() {
        return annealStrategy;
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "OneCycleLR");
        state.put("totalSteps", totalSteps);
        state.put("warmupSteps", warmupSteps);
        state.put("maxLr", maxLr);
        state.put("initialLr", initialLr);
        state.put("minLr", minLr);
        state.put("pctStart", pctStart);
        state.put("divFactor", divFactor);
        state.put("finalDivFactor", finalDivFactor);
        state.put("annealStrategy", annealStrategy.name());
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
        if (schedulerName instanceof String name && !"OneCycleLR".equals(name)) {
            throw new IllegalArgumentException("Checkpoint scheduler mismatch: expected OneCycleLR but got " + name);
        }
        SchedulerValidation.requireIntMatch(state.get("totalSteps"), totalSteps, "OneCycleLR", "totalSteps");
        SchedulerValidation.requireIntMatch(state.get("warmupSteps"), warmupSteps, "OneCycleLR", "warmupSteps");
        SchedulerValidation.requireFloatMatch(state.get("maxLr"), maxLr, "OneCycleLR", "maxLr");
        SchedulerValidation.requireFloatMatch(state.get("initialLr"), initialLr, "OneCycleLR", "initialLr");
        SchedulerValidation.requireFloatMatch(state.get("minLr"), minLr, "OneCycleLR", "minLr");
        SchedulerValidation.requireFloatMatch(state.get("pctStart"), pctStart, "OneCycleLR", "pctStart");
        SchedulerValidation.requireFloatMatch(state.get("divFactor"), divFactor, "OneCycleLR", "divFactor");
        SchedulerValidation.requireFloatMatch(
                state.get("finalDivFactor"), finalDivFactor, "OneCycleLR", "finalDivFactor");
        Object loadedStrategy = state.get("annealStrategy");
        if (loadedStrategy instanceof String name && !annealStrategy.name().equals(name)) {
            throw new IllegalArgumentException(
                    "Invalid OneCycleLR checkpoint payload: annealStrategy mismatch (expected "
                            + annealStrategy.name() + ", got " + name + ")");
        }
        currentStep = SchedulerValidation.readNonNegativeInt(
                state.get("currentStep"), currentStep, "OneCycleLR", "currentStep");
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"), computeLearningRateForStep(currentStep), "OneCycleLR", "currentLr");
        setLearningRate(currentLr);
    }

    private float computeLearningRateForStep(int targetStep) {
        int clampedStep = Math.max(0, Math.min(targetStep, totalSteps));
        if (clampedStep <= warmupSteps) {
            float progress = (float) clampedStep / warmupSteps;
            return anneal(initialLr, maxLr, progress);
        }
        int decaySteps = Math.max(1, totalSteps - warmupSteps);
        float progress = (float) (clampedStep - warmupSteps) / decaySteps;
        return anneal(maxLr, minLr, progress);
    }

    private float anneal(float start, float end, float progress) {
        float clampedProgress = Math.max(0.0f, Math.min(progress, 1.0f));
        if (annealStrategy == AnnealStrategy.LINEAR) {
            return start + (end - start) * clampedProgress;
        }
        return end + 0.5f * (start - end)
                * (1.0f + (float) Math.cos(Math.PI * clampedProgress));
    }
}
