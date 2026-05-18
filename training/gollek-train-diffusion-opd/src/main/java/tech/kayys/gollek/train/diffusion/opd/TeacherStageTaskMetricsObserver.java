package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;

/**
 * Reusable runtime observer that aggregates OPD step diagnostics by task,
 * teacher, stage, and task-stage pair.
 */
public final class TeacherStageTaskMetricsObserver implements DiffusionOpdRuntimeObserver {
    private final Map<String, String> teacherStages;
    private final Map<String, Integer> taskUsage = new LinkedHashMap<>();
    private final Map<String, Double> taskLossSums = new LinkedHashMap<>();
    private final Map<String, Double> taskMeanLoss = new LinkedHashMap<>();
    private final Map<String, Integer> teacherUsage = new LinkedHashMap<>();
    private final Map<String, Double> teacherLossSums = new LinkedHashMap<>();
    private final Map<String, Double> teacherMeanLoss = new LinkedHashMap<>();
    private final Map<String, Integer> stageUsage = new LinkedHashMap<>();
    private final Map<String, Double> stageLossSums = new LinkedHashMap<>();
    private final Map<String, Double> stageMeanLoss = new LinkedHashMap<>();
    private final Map<String, Integer> taskStageUsage = new LinkedHashMap<>();
    private final Map<String, Double> taskStageLossSums = new LinkedHashMap<>();
    private final Map<String, Double> taskStageMeanLoss = new LinkedHashMap<>();

    public TeacherStageTaskMetricsObserver(List<DiffusionTask> tasks) {
        Objects.requireNonNull(tasks, "tasks must not be null");
        this.teacherStages = buildTeacherStageMap(tasks);
        initialize(tasks);
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
        String stageName = teacherStages.getOrDefault(teacherKey, "default");
        String taskStageKey = taskId + "::" + stageName;

        mergeUsageAndLoss(taskUsage, taskLossSums, taskMeanLoss, taskId, stepLoss);
        mergeUsageAndLoss(teacherUsage, teacherLossSums, teacherMeanLoss, teacherKey, stepLoss);
        mergeUsageAndLoss(stageUsage, stageLossSums, stageMeanLoss, stageName, stepLoss);
        mergeUsageAndLoss(taskStageUsage, taskStageLossSums, taskStageMeanLoss, taskStageKey, stepLoss);
    }

    @Override
    public Map<String, Object> summaryMetadata() {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("observedTaskUsage", taskUsage);
        diagnostics.put("observedTaskLossSums", taskLossSums);
        diagnostics.put("observedTaskMeanLoss", taskMeanLoss);
        diagnostics.put("observedTeacherUsage", teacherUsage);
        diagnostics.put("observedTeacherLossSums", teacherLossSums);
        diagnostics.put("observedTeacherMeanLoss", teacherMeanLoss);
        diagnostics.put("observedStageUsage", stageUsage);
        diagnostics.put("observedStageLossSums", stageLossSums);
        diagnostics.put("observedStageMeanLoss", stageMeanLoss);
        diagnostics.put("observedTaskStageUsage", taskStageUsage);
        diagnostics.put("observedTaskStageLossSums", taskStageLossSums);
        diagnostics.put("observedTaskStageMeanLoss", taskStageMeanLoss);
        return Map.copyOf(diagnostics);
    }

    @Override
    public Map<String, Object> roundHistoryMetadata() {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("observedTaskMeanLoss", taskMeanLoss);
        diagnostics.put("observedTeacherMeanLoss", teacherMeanLoss);
        diagnostics.put("observedStageMeanLoss", stageMeanLoss);
        diagnostics.put("observedTaskStageMeanLoss", taskStageMeanLoss);
        return Map.copyOf(diagnostics);
    }

    private void initialize(List<DiffusionTask> tasks) {
        for (DiffusionTask task : tasks) {
            taskUsage.putIfAbsent(task.id(), 0);
            taskLossSums.putIfAbsent(task.id(), 0.0d);
            taskMeanLoss.putIfAbsent(task.id(), 0.0d);
            if (task.teacherBindings().isEmpty()) {
                teacherUsage.putIfAbsent(task.id(), 0);
                teacherLossSums.putIfAbsent(task.id(), 0.0d);
                teacherMeanLoss.putIfAbsent(task.id(), 0.0d);
                stageUsage.putIfAbsent("default", 0);
                stageLossSums.putIfAbsent("default", 0.0d);
                stageMeanLoss.putIfAbsent("default", 0.0d);
                String taskStageKey = task.id() + "::default";
                taskStageUsage.putIfAbsent(taskStageKey, 0);
                taskStageLossSums.putIfAbsent(taskStageKey, 0.0d);
                taskStageMeanLoss.putIfAbsent(taskStageKey, 0.0d);
                continue;
            }
            task.teacherBindings().forEach(binding -> {
                teacherUsage.putIfAbsent(binding.teacherKey(), 0);
                teacherLossSums.putIfAbsent(binding.teacherKey(), 0.0d);
                teacherMeanLoss.putIfAbsent(binding.teacherKey(), 0.0d);
                stageUsage.putIfAbsent(binding.stageName(), 0);
                stageLossSums.putIfAbsent(binding.stageName(), 0.0d);
                stageMeanLoss.putIfAbsent(binding.stageName(), 0.0d);
                String taskStageKey = task.id() + "::" + binding.stageName();
                taskStageUsage.putIfAbsent(taskStageKey, 0);
                taskStageLossSums.putIfAbsent(taskStageKey, 0.0d);
                taskStageMeanLoss.putIfAbsent(taskStageKey, 0.0d);
            });
        }
    }

    private static Map<String, String> buildTeacherStageMap(List<DiffusionTask> tasks) {
        Map<String, String> mapping = new LinkedHashMap<>();
        for (DiffusionTask task : tasks) {
            if (task.teacherBindings().isEmpty()) {
                mapping.putIfAbsent(task.id(), "default");
                continue;
            }
            task.teacherBindings().forEach(binding ->
                    mapping.putIfAbsent(binding.teacherKey(), binding.stageName()));
        }
        return Map.copyOf(mapping);
    }

    private static void mergeUsageAndLoss(
            Map<String, Integer> usage,
            Map<String, Double> lossSums,
            Map<String, Double> meanLoss,
            String key,
            double stepLoss) {
        usage.merge(key, 1, Integer::sum);
        lossSums.merge(key, stepLoss, Double::sum);
        int steps = Math.max(1, usage.getOrDefault(key, 1));
        double lossSum = lossSums.getOrDefault(key, 0.0d);
        meanLoss.put(key, lossSum / steps);
    }
}
