package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Preset helpers for stage-aware teacher routing in diffusion training.
 */
public final class DiffusionTeacherBindings {

    private DiffusionTeacherBindings() {
    }

    public static List<DiffusionTeacherBinding> splitEarlyLate(
            int totalSteps,
            String earlyTeacherKey,
            String lateTeacherKey) {
        return splitEarlyLate(totalSteps, earlyTeacherKey, lateTeacherKey, 1.0d, 1.0d);
    }

    public static List<DiffusionTeacherBinding> splitEarlyLate(
            int totalSteps,
            String earlyTeacherKey,
            String lateTeacherKey,
            double earlyLossWeight,
            double lateLossWeight) {
        int pivot = clampPivot(totalSteps, totalSteps / 2);
        return List.of(
                DiffusionTeacherBinding.weightedStage(
                        earlyTeacherKey,
                        0,
                        pivot,
                        "early",
                        earlyLossWeight),
                DiffusionTeacherBinding.weightedStage(
                        lateTeacherKey,
                        pivot,
                        totalSteps,
                        "late",
                        lateLossWeight));
    }

    public static List<DiffusionTeacherBinding> splitEarlyMidLate(
            int totalSteps,
            String earlyTeacherKey,
            String midTeacherKey,
            String lateTeacherKey) {
        return splitEarlyMidLate(
                totalSteps,
                earlyTeacherKey,
                midTeacherKey,
                lateTeacherKey,
                1.0d,
                1.0d,
                1.0d);
    }

    public static List<DiffusionTeacherBinding> splitEarlyMidLate(
            int totalSteps,
            String earlyTeacherKey,
            String midTeacherKey,
            String lateTeacherKey,
            double earlyLossWeight,
            double midLossWeight,
            double lateLossWeight) {
        if (totalSteps < 3) {
            throw new IllegalArgumentException("totalSteps must be >= 3 for early/mid/late split");
        }
        int earlyEnd = clampPivot(totalSteps, Math.max(1, totalSteps / 3));
        int lateStart = clampPivot(totalSteps, Math.max(earlyEnd + 1, (2 * totalSteps) / 3));
        return List.of(
                DiffusionTeacherBinding.weightedStage(
                        earlyTeacherKey,
                        0,
                        earlyEnd,
                        "early",
                        earlyLossWeight),
                DiffusionTeacherBinding.weightedStage(
                        midTeacherKey,
                        earlyEnd,
                        lateStart,
                        "mid",
                        midLossWeight),
                DiffusionTeacherBinding.weightedStage(
                        lateTeacherKey,
                        lateStart,
                        totalSteps,
                        "late",
                        lateLossWeight));
    }

    private static int clampPivot(int totalSteps, int pivot) {
        if (totalSteps < 2) {
            throw new IllegalArgumentException("totalSteps must be >= 2 for stage partitioning");
        }
        return Math.max(1, Math.min(totalSteps - 1, pivot));
    }
}
