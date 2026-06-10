package tech.kayys.gollek.ml.train;

import java.util.List;

/**
 * Converts wall-clock trainer overhead into targeted runtime-performance recommendations.
 */
final class TrainingReportRuntimeWallClockAdvisor {
    private TrainingReportRuntimeWallClockAdvisor() {
    }

    static List<TrainingReportRecommendation> recommendations(TrainingReportRuntimeProfile profile) {
        TrainingReportRuntimeProfile.WallClock wallClock =
                profile == null ? TrainingReportRuntimeProfile.WallClock.empty() : profile.wallClock();
        TrainingReportRuntimeWallClockAssessment assessment =
                TrainingReportRuntimeWallClockAssessment.assess(wallClock);
        if (!assessment.available() || !assessment.overheadDetected()) {
            return List.of();
        }
        return List.of(new TrainingReportRecommendation(
                assessment.priority(),
                assessment.category(),
                assessment.severity(),
                "runtime_profile.wall_clock.overhead",
                "Reduce unprofiled trainer overhead in `" + assessment.scopeKey() + "`",
                "The wall-clock profiler reports trainer scope time that is not explained by measured sub-phases.",
                actions(assessment.scope()),
                assessment.recommendationEvidence(wallClock.totalMillis(), wallClock.scopeCount())));
    }

    private static List<String> actions(TrainingReportRuntimeWallClockAssessment.Scope scope) {
        return switch (scope) {
            case OPTIMIZER_STEP -> List.of(
                    "Inspect optimizer-step glue around diagnostics, scheduler callbacks, and state updates before changing tensor kernels.",
                    "Compare `runtimeProfile.wall.optimizerStep.profiledMillis` and `overheadMillis` after disabling optional diagnostics.",
                    "Add a focused optimizer-step benchmark when overhead remains high after diagnostic sampling.");
            case VALIDATION_BATCH -> List.of(
                    "Inspect validation batch orchestration, metric updates, and no-grad scope handling before changing model code.",
                    "Reduce validation frequency or metric detail when validation wall-clock overhead dominates short runs.",
                    "Compare validation wall overhead before and after caching validation-only setup.");
            case TRAIN_BATCH -> List.of(
                    "Inspect trainer batch orchestration around guards, metric hooks, and framework callbacks.",
                    "Move static per-batch setup out of the hot training loop when wall-clock overhead remains high.",
                    "Add a focused train-batch smoke benchmark so framework overhead regressions fail early.");
            case NONE -> List.of(
                    "Inspect trainer glue code around the reported wall-clock scope.",
                    "Compare wall-clock overhead across two short profiling runs before optimizing unrelated code.",
                    "Add a focused benchmark for the scope when overhead is stable.");
        };
    }
}
