package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Compact end-of-rollout summary for recursive reasoning runs.
 */
public record RecursiveReasoningRolloutSummary(
        int exploredTrajectoryCount,
        int completedTrajectoryCount,
        String selectedStateId,
        Map<String, Object> metadata) {

    public RecursiveReasoningRolloutSummary {
        exploredTrajectoryCount = Math.max(0, exploredTrajectoryCount);
        completedTrajectoryCount = Math.max(0, completedTrajectoryCount);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
