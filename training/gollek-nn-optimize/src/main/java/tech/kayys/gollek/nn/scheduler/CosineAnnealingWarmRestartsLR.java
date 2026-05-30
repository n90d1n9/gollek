package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Cosine annealing scheduler with warm restarts.
 *
 * <p>
 * Each cycle anneals from the optimizer's initial learning rate down to
 * {@code minLr}. At the next step the scheduler restarts at the initial
 * learning rate and optionally lengthens the cycle by {@code cycleMultiplier}.
 * This mirrors the SGDR training pattern while remaining simple to checkpoint.
 */
public final class CosineAnnealingWarmRestartsLR extends LRScheduler {

    private final float initialLr;
    private final float minLr;
    private final int firstCycleSteps;
    private final int cycleMultiplier;
    private int currentStep;
    private int cycleIndex;
    private int cycleStep;
    private int cycleLength;
    private float currentLr;

    public CosineAnnealingWarmRestartsLR(
            Optimizer optimizer,
            int firstCycleSteps,
            int cycleMultiplier,
            float minLr) {
        super(optimizer);
        if (firstCycleSteps <= 0) {
            throw new IllegalArgumentException("firstCycleSteps must be positive, got: " + firstCycleSteps);
        }
        if (cycleMultiplier < 1) {
            throw new IllegalArgumentException("cycleMultiplier must be >= 1, got: " + cycleMultiplier);
        }
        this.initialLr = SchedulerValidation.learningRate(optimizer.learningRate(), "initialLr");
        this.minLr = SchedulerValidation.nonNegative(minLr, "minLr");
        if (this.minLr > this.initialLr) {
            throw new IllegalArgumentException(
                    "minLr must be <= initialLr, got: " + this.minLr + " > " + this.initialLr);
        }
        this.firstCycleSteps = firstCycleSteps;
        this.cycleMultiplier = cycleMultiplier;
        this.cycleLength = firstCycleSteps;
        this.currentLr = initialLr;
    }

    @Override
    public void step() {
        currentLr = computeLearningRate(cycleStep, cycleLength);
        setLearningRate(currentLr);
        currentStep++;
        cycleStep++;
        if (cycleStep >= cycleLength) {
            cycleStep = 0;
            cycleIndex++;
            cycleLength = nextCycleLength(cycleLength);
        }
    }

    @Override
    public float getLr() {
        return currentLr;
    }

    public int currentStep() {
        return currentStep;
    }

    public int cycleIndex() {
        return cycleIndex;
    }

    public int cycleStep() {
        return cycleStep;
    }

    public int cycleLength() {
        return cycleLength;
    }

    public int firstCycleSteps() {
        return firstCycleSteps;
    }

    public int cycleMultiplier() {
        return cycleMultiplier;
    }

    public float initialLr() {
        return initialLr;
    }

    public float minLr() {
        return minLr;
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "CosineAnnealingWarmRestartsLR");
        state.put("initialLr", initialLr);
        state.put("minLr", minLr);
        state.put("firstCycleSteps", firstCycleSteps);
        state.put("cycleMultiplier", cycleMultiplier);
        state.put("currentStep", currentStep);
        state.put("cycleIndex", cycleIndex);
        state.put("cycleStep", cycleStep);
        state.put("cycleLength", cycleLength);
        state.put("currentLr", currentLr);
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"CosineAnnealingWarmRestartsLR".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected CosineAnnealingWarmRestartsLR but got " + name);
        }
        SchedulerValidation.requireFloatMatch(
                state.get("initialLr"), initialLr, "CosineAnnealingWarmRestartsLR", "initialLr");
        SchedulerValidation.requireFloatMatch(
                state.get("minLr"), minLr, "CosineAnnealingWarmRestartsLR", "minLr");
        SchedulerValidation.requireIntMatch(
                state.get("firstCycleSteps"), firstCycleSteps,
                "CosineAnnealingWarmRestartsLR", "firstCycleSteps");
        SchedulerValidation.requireIntMatch(
                state.get("cycleMultiplier"), cycleMultiplier,
                "CosineAnnealingWarmRestartsLR", "cycleMultiplier");
        currentStep = SchedulerValidation.readNonNegativeInt(
                state.get("currentStep"), currentStep,
                "CosineAnnealingWarmRestartsLR", "currentStep");
        cycleIndex = SchedulerValidation.readNonNegativeInt(
                state.get("cycleIndex"), cycleIndex,
                "CosineAnnealingWarmRestartsLR", "cycleIndex");
        cycleStep = SchedulerValidation.readNonNegativeInt(
                state.get("cycleStep"), cycleStep,
                "CosineAnnealingWarmRestartsLR", "cycleStep");
        cycleLength = SchedulerValidation.readNonNegativeInt(
                state.get("cycleLength"), cycleLength,
                "CosineAnnealingWarmRestartsLR", "cycleLength");
        if (cycleLength <= 0 || cycleStep >= cycleLength) {
            throw new IllegalArgumentException(
                    "Invalid CosineAnnealingWarmRestartsLR checkpoint payload: "
                            + "cycleStep must be in [0, cycleLength)");
        }
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"),
                computeLearningRate(cycleStep, cycleLength),
                "CosineAnnealingWarmRestartsLR",
                "currentLr");
        setLearningRate(currentLr);
    }

    private float computeLearningRate(int targetCycleStep, int targetCycleLength) {
        if (targetCycleLength <= 1) {
            return initialLr;
        }
        double progress = Math.max(0, Math.min(targetCycleStep, targetCycleLength - 1))
                / (double) (targetCycleLength - 1);
        double cosineDecay = 0.5 * (1.0 + Math.cos(Math.PI * progress));
        return (float) (minLr + (initialLr - minLr) * cosineDecay);
    }

    private int nextCycleLength(int currentCycleLength) {
        if (cycleMultiplier == 1) {
            return currentCycleLength;
        }
        long next = (long) currentCycleLength * cycleMultiplier;
        return next > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) next;
    }

    @Override
    public String toString() {
        return "CosineAnnealingWarmRestartsLR(firstCycleSteps=" + firstCycleSteps
                + ", cycleMultiplier=" + cycleMultiplier
                + ", minLr=" + minLr + ")";
    }
}
