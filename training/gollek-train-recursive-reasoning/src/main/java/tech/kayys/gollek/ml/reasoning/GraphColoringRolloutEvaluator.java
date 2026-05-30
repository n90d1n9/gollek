package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Scores recursive-reasoning rollout samples as graph-coloring candidates.
 */
public final class GraphColoringRolloutEvaluator {
    private GraphColoringRolloutEvaluator() {
    }

    public static GraphColoringRolloutEvaluationReport evaluate(
            GraphColoringProblem problem,
            RecursiveReasoningRolloutResult rollout,
            DiscreteStateTokenDecoder decoder) {
        return evaluate(problem, rollout, decoder, -1);
    }

    public static GraphColoringRolloutEvaluationReport evaluate(
            GraphColoringProblem problem,
            RecursiveReasoningRolloutResult rollout,
            DiscreteStateTokenDecoder decoder,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        Objects.requireNonNull(rollout, "rollout must not be null");
        Objects.requireNonNull(decoder, "decoder must not be null");
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        DiscreteRolloutEvaluationReport discreteReport = DiscreteRolloutEvaluator.evaluate(
                rollout,
                decoder,
                prediction -> GraphColoringBenchmark.evaluateTokensAsDiscrete(problem, prediction.tokens()),
                knownSolutionCount);
        List<GraphColoringTrajectoryEvaluation> evaluations = new ArrayList<>();
        for (DiscreteTrajectoryEvaluation trajectoryEvaluation : discreteReport.evaluations()) {
            int[] tokens = trajectoryEvaluation.tokens();
            GraphColoringTokenDecodeResult decoded = GraphColoringTokenCodec.decodeSolution(
                    problem.nodeCount(),
                    problem.colorCount(),
                    tokens);
            GraphColoringEvaluation evaluation = GraphColoringBenchmark.evaluateTokens(problem, tokens);
            evaluations.add(new GraphColoringTrajectoryEvaluation(
                    trajectoryEvaluation.trajectoryIndex(),
                    trajectoryEvaluation.sampleIndex(),
                    trajectoryEvaluation.stateId(),
                    tokens,
                    decoded,
                    evaluation,
                    trajectoryEvaluation.metadata()));
        }

        return new GraphColoringRolloutEvaluationReport(
                problem,
                evaluations,
                discreteReport.selectedTrajectoryIndex(),
                GraphColoringBenchmark.toGraphColoringCoverageReport(discreteReport.coverage()),
                discreteReport.metadata());
    }
}
