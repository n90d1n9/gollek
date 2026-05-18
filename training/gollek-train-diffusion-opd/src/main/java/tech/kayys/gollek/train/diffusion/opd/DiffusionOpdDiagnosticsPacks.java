package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;

/**
 * Convenience factories for common DiffusionOPD diagnostics bundles.
 */
public final class DiffusionOpdDiagnosticsPacks {

    private DiffusionOpdDiagnosticsPacks() {
    }

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
