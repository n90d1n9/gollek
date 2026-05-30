package tech.kayys.gollek.train.diffusion.opd;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBinding;

/**
 * Owns adaptive stage-weighting policy and round-history shaping.
 *
 * <p>This helper keeps the trainer's stage-factor policy, mean-loss rollups,
 * and round-history row assembly together so the main trainer session can stay
 * focused on rollout flow.
 */
final class DefaultDiffusionOpdTrainerAdaptiveSupport {

    private DefaultDiffusionOpdTrainerAdaptiveSupport() {
    }

    static Map<String, Double> initializeAdaptiveTaskStageFactors(List<DiffusionTask> tasks) {
        Map<String, Double> factors = new LinkedHashMap<>();
        for (DiffusionTask task : tasks) {
            if (task.teacherBindings().isEmpty()) {
                factors.putIfAbsent(taskStageKey(task.id(), "default"), 1.0d);
                continue;
            }
            for (DiffusionTeacherBinding binding : task.teacherBindings()) {
                factors.putIfAbsent(taskStageKey(task.id(), binding.stageName()), 1.0d);
            }
        }
        return factors;
    }

    static double effectiveStageWeight(
            boolean adaptiveStageWeighting,
            String taskStageKey,
            StageAwareTeacherSelector.ResolvedTeacher resolvedTeacher,
            Map<String, Double> adaptiveTaskStageFactors) {
        double adaptiveFactor = adaptiveStageWeighting
                ? adaptiveTaskStageFactors.getOrDefault(taskStageKey, 1.0d)
                : 1.0d;
        return resolvedTeacher.lossWeight() * adaptiveFactor;
    }

    static void adaptStageWeights(
            boolean adaptiveStageWeighting,
            double adaptiveStageWeightMomentum,
            double adaptiveStageWeightMinFactor,
            double adaptiveStageWeightMaxFactor,
            Map<String, Double> roundBaseStageLoss,
            Map<String, Integer> roundStageSteps,
            Map<String, Double> adaptiveTaskStageFactors) {
        if (!adaptiveStageWeighting || roundBaseStageLoss.isEmpty()) {
            return;
        }
        Map<String, Double> stageMeanLoss = meanStageLoss(roundBaseStageLoss, roundStageSteps);
        double globalMean = 0.0d;
        for (double meanLoss : stageMeanLoss.values()) {
            globalMean += meanLoss;
        }
        globalMean /= stageMeanLoss.size();
        if (!Double.isFinite(globalMean) || globalMean <= 0.0d) {
            return;
        }
        for (Map.Entry<String, Double> entry : stageMeanLoss.entrySet()) {
            double currentFactor = adaptiveTaskStageFactors.getOrDefault(entry.getKey(), 1.0d);
            double targetFactor = clamp(entry.getValue() / globalMean, adaptiveStageWeightMinFactor, adaptiveStageWeightMaxFactor);
            double blendedFactor = (adaptiveStageWeightMomentum * currentFactor)
                    + ((1.0d - adaptiveStageWeightMomentum) * targetFactor);
            adaptiveTaskStageFactors.put(
                    entry.getKey(),
                    clamp(blendedFactor, adaptiveStageWeightMinFactor, adaptiveStageWeightMaxFactor));
        }
    }

    static void recordRoundHistory(
            List<Map<String, Object>> roundHistory,
            Map<String, Object> roundHistoryMetadata,
            Map<String, Object> runtimeRoundHistoryMetadata,
            int round,
            double meanLoss,
            int optimizationSteps,
            Map<String, Double> adaptiveTaskStageFactors,
            Map<String, Double> roundBaseStageLoss,
            Map<String, Integer> roundStageSteps) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("round", round);
        row.put("meanLoss", meanLoss);
        row.put("optimizationSteps", optimizationSteps);
        row.put("adaptiveTaskStageFactors", Map.copyOf(adaptiveTaskStageFactors));
        row.put("adaptiveStageFactors", aggregateStageFactors(adaptiveTaskStageFactors));
        row.put("roundBaseStageMeanLoss", meanStageLoss(roundBaseStageLoss, roundStageSteps));
        row.putAll(roundHistoryMetadata);
        row.putAll(runtimeRoundHistoryMetadata);
        roundHistory.add(Map.copyOf(row));
        roundHistory.sort(Comparator.comparingInt(entry -> ((Number) entry.get("round")).intValue()));
    }

    static Map<String, Double> aggregateStageFactors(Map<String, Double> adaptiveTaskStageFactors) {
        Map<String, Double> sums = new LinkedHashMap<>();
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : adaptiveTaskStageFactors.entrySet()) {
            String stageName = stageNameFromTaskStageKey(entry.getKey());
            sums.merge(stageName, entry.getValue(), Double::sum);
            counts.merge(stageName, 1, Integer::sum);
        }
        Map<String, Double> aggregate = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : sums.entrySet()) {
            int count = Math.max(1, counts.getOrDefault(entry.getKey(), 1));
            aggregate.put(entry.getKey(), entry.getValue() / count);
        }
        return Map.copyOf(aggregate);
    }

    static Map<String, Double> meanStageLoss(
            Map<String, Double> stageLoss,
            Map<String, Integer> stageSteps) {
        Map<String, Double> meanLoss = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : stageLoss.entrySet()) {
            int steps = Math.max(1, stageSteps.getOrDefault(entry.getKey(), 1));
            meanLoss.put(entry.getKey(), entry.getValue() / steps);
        }
        return Map.copyOf(meanLoss);
    }

    static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    static String taskStageKey(String taskId, String stageName) {
        return taskId + "::" + stageName;
    }

    private static String stageNameFromTaskStageKey(String taskStageKey) {
        int separator = taskStageKey.indexOf("::");
        return separator >= 0 ? taskStageKey.substring(separator + 2) : taskStageKey;
    }
}
