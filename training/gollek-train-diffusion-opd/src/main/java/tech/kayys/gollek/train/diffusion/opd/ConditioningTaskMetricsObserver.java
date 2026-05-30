package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;

/**
 * Reusable runtime observer that records per-task conditioning lane usage and
 * loss aggregates into the standard DiffusionOPD summary/history artifacts.
 *
 * <p>This observer owns conditioning-specific diagnostics only. General
 * task/teacher/stage loss aggregation is handled by
 * {@link TeacherStageTaskMetricsObserver}.
 */
public final class ConditioningTaskMetricsObserver implements DiffusionOpdRuntimeObserver {
    private final Map<String, String> taskConditioningModes = new LinkedHashMap<>();
    private final Map<String, String> taskConditioningFixtureBaseDirs = new LinkedHashMap<>();
    private final Map<String, Integer> taskConditioningResolveCounts = new LinkedHashMap<>();
    private final Map<String, Integer> taskConditioningLossSteps = new LinkedHashMap<>();
    private final Map<String, Double> taskConditioningLossSums = new LinkedHashMap<>();
    private final Map<String, Double> taskConditioningMeanLoss = new LinkedHashMap<>();
    private final String taskConditioningFixtures;

    /**
     * Creates the conditioning-oriented observer with optional preseeded task modes, fixture roots,
     * and a human-readable fixture summary string for downstream report metadata.
     */
    public ConditioningTaskMetricsObserver(
            Map<String, String> initialModes,
            Map<String, String> fixtureBaseDirs,
            String fixturesSummary) {
        Objects.requireNonNull(initialModes, "initialModes must not be null")
                .forEach((taskId, mode) -> {
                    taskConditioningModes.put(taskId, mode);
                    taskConditioningResolveCounts.put(taskId, 0);
                    taskConditioningLossSteps.put(taskId, 0);
                    taskConditioningLossSums.put(taskId, 0.0d);
                    taskConditioningMeanLoss.put(taskId, 0.0d);
                });
        Objects.requireNonNull(fixtureBaseDirs, "fixtureBaseDirs must not be null")
                .forEach(taskConditioningFixtureBaseDirs::put);
        this.taskConditioningFixtures = fixturesSummary == null ? "" : fixturesSummary;
    }

    @Override
    public void onConditioningResolved(
            DiffusionOpdSession session,
            int round,
            DiffusionTask task,
            DiffusionPromptSample sample,
            Tensor conditioning) {
        String taskId = task.id();
        String mode = taskConditioningModes.getOrDefault(taskId, "synthetic");
        taskConditioningModes.put(taskId, mode);
        taskConditioningResolveCounts.merge(taskId, 1, Integer::sum);
    }

    @Override
    public void onStep(
            DiffusionOpdSession session,
            int round,
            DiffusionTask task,
            int timestep,
            String teacherKey,
            double stepLoss) {
        String taskId = task.id();
        taskConditioningLossSteps.merge(taskId, 1, Integer::sum);
        taskConditioningLossSums.merge(taskId, stepLoss, Double::sum);
        int steps = Math.max(1, taskConditioningLossSteps.getOrDefault(taskId, 1));
        double lossSum = taskConditioningLossSums.getOrDefault(taskId, 0.0d);
        taskConditioningMeanLoss.put(taskId, lossSum / steps);
    }

    @Override
    public Map<String, Object> summaryMetadata() {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("conditioningTaskModes", taskConditioningModes);
        diagnostics.put("conditioningTaskFixtureBaseDirs", taskConditioningFixtureBaseDirs);
        diagnostics.put("conditioningTaskFixtures", taskConditioningFixtures);
        diagnostics.put("conditioningTaskResolveCounts", taskConditioningResolveCounts);
        diagnostics.put("conditioningTaskLossSteps", taskConditioningLossSteps);
        diagnostics.put("conditioningTaskLossSums", taskConditioningLossSums);
        diagnostics.put("conditioningTaskMeanLoss", taskConditioningMeanLoss);
        return Map.copyOf(diagnostics);
    }

    @Override
    public Map<String, Object> roundHistoryMetadata() {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("conditioningTaskModes", taskConditioningModes);
        diagnostics.put("conditioningTaskResolveCounts", taskConditioningResolveCounts);
        diagnostics.put("conditioningTaskMeanLoss", taskConditioningMeanLoss);
        return Map.copyOf(diagnostics);
    }
}
