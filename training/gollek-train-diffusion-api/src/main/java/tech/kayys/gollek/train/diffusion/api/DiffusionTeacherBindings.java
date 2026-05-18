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

    public static List<DiffusionTeacherBinding> splitEarlyLate(
            DiffusionScheduler scheduler,
            String earlyTeacherKey,
            String lateTeacherKey) {
        return splitEarlyLate(scheduler, earlyTeacherKey, lateTeacherKey, 1.0d, 1.0d);
    }

    public static List<DiffusionTeacherBinding> splitEarlyLate(
            DiffusionScheduler scheduler,
            String earlyTeacherKey,
            String lateTeacherKey,
            double earlyLossWeight,
            double lateLossWeight) {
        int[] timesteps = validateTimesteps(scheduler);
        int pivot = findBoundaryIndex(timesteps, 0.50d);
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
                        timesteps.length,
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

    public static List<DiffusionTeacherBinding> splitEarlyMidLate(
            DiffusionScheduler scheduler,
            String earlyTeacherKey,
            String midTeacherKey,
            String lateTeacherKey) {
        return splitEarlyMidLate(
                scheduler,
                earlyTeacherKey,
                midTeacherKey,
                lateTeacherKey,
                1.0d,
                1.0d,
                1.0d);
    }

    public static List<DiffusionTeacherBinding> splitEarlyMidLate(
            DiffusionScheduler scheduler,
            String earlyTeacherKey,
            String midTeacherKey,
            String lateTeacherKey,
            double earlyLossWeight,
            double midLossWeight,
            double lateLossWeight) {
        int[] timesteps = validateTimesteps(scheduler);
        if (timesteps.length < 3) {
            throw new IllegalArgumentException("scheduler must expose at least 3 timesteps for early/mid/late split");
        }
        int earlyEnd = findBoundaryIndex(timesteps, 1.0d / 3.0d);
        int lateStart = Math.max(earlyEnd + 1, findBoundaryIndex(timesteps, 2.0d / 3.0d));
        lateStart = Math.min(timesteps.length - 1, lateStart);
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
                        timesteps.length,
                        "late",
                        lateLossWeight));
    }

    public static String partitionSummary(DiffusionScheduler scheduler) {
        int[] timesteps = validateTimesteps(scheduler);
        double[] masses = stepMasses(timesteps);
        StringBuilder summary = new StringBuilder("scheduler-aware[");
        for (int i = 0; i < timesteps.length; i++) {
            if (i > 0) {
                summary.append(", ");
            }
            summary.append("step").append(i)
                    .append("=t").append(timesteps[i])
                    .append("/m").append(String.format(java.util.Locale.ROOT, "%.1f", masses[i]));
        }
        return summary.append("]").toString();
    }

    private static int clampPivot(int totalSteps, int pivot) {
        if (totalSteps < 2) {
            throw new IllegalArgumentException("totalSteps must be >= 2 for stage partitioning");
        }
        return Math.max(1, Math.min(totalSteps - 1, pivot));
    }

    private static int[] validateTimesteps(DiffusionScheduler scheduler) {
        if (scheduler == null) {
            throw new IllegalArgumentException("scheduler must not be null");
        }
        int[] timesteps = scheduler.timesteps();
        if (timesteps == null || timesteps.length < 2) {
            throw new IllegalArgumentException("scheduler must expose at least 2 timesteps");
        }
        return timesteps;
    }

    private static int findBoundaryIndex(int[] timesteps, double targetFraction) {
        double[] masses = stepMasses(timesteps);
        double totalMass = 0.0d;
        for (double mass : masses) {
            totalMass += mass;
        }
        double targetMass = totalMass * targetFraction;
        double cumulative = 0.0d;
        for (int i = 0; i < masses.length - 1; i++) {
            cumulative += masses[i];
            if (cumulative >= targetMass) {
                return i + 1;
            }
        }
        return timesteps.length - 1;
    }

    private static double[] stepMasses(int[] timesteps) {
        double[] masses = new double[timesteps.length];
        for (int i = 0; i < timesteps.length; i++) {
            int current = timesteps[i];
            int next = i == timesteps.length - 1 ? 0 : timesteps[i + 1];
            masses[i] = Math.max(1.0d, Math.abs((double) current - next));
        }
        return masses;
    }
}
