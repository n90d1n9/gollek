package tech.kayys.gollek.train.diffusion.api;

import java.util.Objects;

/**
 * Binds a teacher identifier to a timestep interval for stage-aware routing.
 *
 * <p>This is the Java-first counterpart to stage-specialized denoisers inspired
 * by eDiff-I, while still fitting the DiffusionOPD training loop.
 */
public record DiffusionTeacherBinding(
        String teacherKey,
        int startStepInclusive,
        int endStepExclusive,
        String stageName,
        double lossWeight) {

    public DiffusionTeacherBinding {
        teacherKey = Objects.requireNonNull(teacherKey, "teacherKey must not be null");
        stageName = stageName == null || stageName.isBlank() ? "unnamed-stage" : stageName;
        if (startStepInclusive < 0) {
            throw new IllegalArgumentException("startStepInclusive must be >= 0");
        }
        if (endStepExclusive <= startStepInclusive) {
            throw new IllegalArgumentException("endStepExclusive must be greater than startStepInclusive");
        }
        if (!Double.isFinite(lossWeight) || lossWeight <= 0.0d) {
            throw new IllegalArgumentException("lossWeight must be finite and > 0");
        }
    }

    public boolean matchesStep(int stepIndex) {
        return stepIndex >= startStepInclusive && stepIndex < endStepExclusive;
    }

    public static DiffusionTeacherBinding full(String teacherKey) {
        return new DiffusionTeacherBinding(teacherKey, 0, Integer.MAX_VALUE, "all", 1.0d);
    }

    public static DiffusionTeacherBinding stage(
            String teacherKey,
            int startStepInclusive,
            int endStepExclusive,
            String stageName) {
        return new DiffusionTeacherBinding(teacherKey, startStepInclusive, endStepExclusive, stageName, 1.0d);
    }

    public static DiffusionTeacherBinding weightedStage(
            String teacherKey,
            int startStepInclusive,
            int endStepExclusive,
            String stageName,
            double lossWeight) {
        return new DiffusionTeacherBinding(
                teacherKey,
                startStepInclusive,
                endStepExclusive,
                stageName,
                lossWeight);
    }
}
