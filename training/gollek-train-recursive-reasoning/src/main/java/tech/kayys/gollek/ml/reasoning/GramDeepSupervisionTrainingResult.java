package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Result of a backend-neutral GRAM deep-supervision reference training pass.
 */
public record GramDeepSupervisionTrainingResult(
        RecursiveReasoningTrajectory trajectory,
        List<GramVariationalTransitionResult> transitions,
        GramObjectiveBreakdown objective,
        TrainingSummary summary,
        Map<String, Object> metadata) {

    public GramDeepSupervisionTrainingResult {
        trajectory = Objects.requireNonNull(trajectory, "trajectory must not be null");
        transitions = List.copyOf(Objects.requireNonNull(transitions, "transitions must not be null"));
        objective = Objects.requireNonNull(objective, "objective must not be null");
        summary = Objects.requireNonNull(summary, "summary must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
