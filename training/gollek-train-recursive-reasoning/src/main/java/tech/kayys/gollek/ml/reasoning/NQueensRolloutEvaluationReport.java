package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * N-Queens benchmark report for a full recursive-reasoning rollout.
 */
public record NQueensRolloutEvaluationReport(
        NQueensProblem problem,
        List<NQueensTrajectoryEvaluation> evaluations,
        int selectedTrajectoryIndex,
        NQueensCoverageReport coverage,
        Map<String, Object> metadata) {

    public NQueensRolloutEvaluationReport {
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

    public NQueensTrajectoryEvaluation selectedEvaluation() {
        return evaluations.get(selectedTrajectoryIndex);
    }
}
