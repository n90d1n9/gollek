package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;

/**
 * Convenience factories for common DiffusionOPD diagnostics bundles.
 *
 * <p>This is the public assembly point for the reusable runtime-observer
 * cluster. It keeps callers off the individual observer wiring details while
 * composing the maintained standard diagnostics pack.
 */
public final class DiffusionOpdDiagnosticsPacks {

    private DiffusionOpdDiagnosticsPacks() {
    }

    /**
     * Builds the maintained default observer pack for task-level diffusion diagnostics.
     *
     * <p>The returned observers cover both conditioning-oriented diagnostics and the broader
     * teacher/stage/task loss breakdown used by OPD training runs.
     */
    public static List<DiffusionOpdRuntimeObserver> standardTaskDiagnostics(
            List<DiffusionTask> tasks,
            Map<String, String> conditioningModes,
            Map<String, String> conditioningFixtureBaseDirs,
            String conditioningFixturesSummary) {
        return List.of(
                new ConditioningTaskMetricsObserver(
                        conditioningModes,
                        conditioningFixtureBaseDirs,
                        conditioningFixturesSummary),
                new TeacherStageTaskMetricsObserver(tasks));
    }
}
