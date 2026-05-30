package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Objects;

/**
 * Full result of a reference recursive reasoning rollout.
 */
public record RecursiveReasoningRolloutResult(
        List<RecursiveReasoningTrajectory> trajectories,
        int selectedTrajectoryIndex,
        RecursiveReasoningRolloutSummary summary) {

    public RecursiveReasoningRolloutResult {
        trajectories = List.copyOf(Objects.requireNonNull(trajectories, "trajectories must not be null"));
        if (trajectories.isEmpty()) {
            throw new IllegalArgumentException("trajectories must not be empty");
        }
        if (selectedTrajectoryIndex < 0 || selectedTrajectoryIndex >= trajectories.size()) {
            throw new IllegalArgumentException(
                    "selectedTrajectoryIndex must point to a trajectory but was " + selectedTrajectoryIndex);
        }
        summary = Objects.requireNonNull(summary, "summary must not be null");
    }

    public RecursiveReasoningTrajectory selectedTrajectory() {
        return trajectories.get(selectedTrajectoryIndex);
    }
}
