package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Benchmark-neutral evaluation report for a recursive rollout with discrete outputs.
 */
public record DiscreteRolloutEvaluationReport(
        List<DiscreteTrajectoryEvaluation> evaluations,
        int selectedTrajectoryIndex,
        DiscreteTokenCoverageReport coverage,
        Map<String, Object> metadata) {

    public DiscreteRolloutEvaluationReport {
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

    public DiscreteTrajectoryEvaluation selectedEvaluation() {
        return evaluations.get(selectedTrajectoryIndex);
    }
}
