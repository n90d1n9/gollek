package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Graph-coloring benchmark report for a full recursive rollout.
 */
public record GraphColoringRolloutEvaluationReport(
        GraphColoringProblem problem,
        List<GraphColoringTrajectoryEvaluation> evaluations,
        int selectedTrajectoryIndex,
        GraphColoringCoverageReport coverage,
        Map<String, Object> metadata) {

    public GraphColoringRolloutEvaluationReport {
        problem = Objects.requireNonNull(problem, "problem must not be null");
        evaluations = List.copyOf(Objects.requireNonNull(evaluations, "evaluations must not be null"));
        if (evaluations.isEmpty()) {
            throw new IllegalArgumentException("evaluations must not be empty");
        }
        if (selectedTrajectoryIndex < 0 || selectedTrajectoryIndex >= evaluations.size()) {
            throw new IllegalArgumentException(
                    "selectedTrajectoryIndex must point to an evaluation but was " + selectedTrajectoryIndex);
        }
        coverage = Objects.requireNonNull(coverage, "coverage must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public GraphColoringTrajectoryEvaluation selectedEvaluation() {
        return evaluations.get(selectedTrajectoryIndex);
    }
}
