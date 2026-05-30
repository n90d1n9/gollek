package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Scores recursive-reasoning rollout samples as N-Queens candidate boards.
 */
public final class NQueensRolloutEvaluator {
    private NQueensRolloutEvaluator() {
    }

    public static NQueensRolloutEvaluationReport evaluate(
            NQueensProblem problem,
            RecursiveReasoningRolloutResult rollout,
            NQueensStateTokenDecoder decoder) {
        return evaluate(problem, rollout, decoder, -1);
    }

    public static NQueensRolloutEvaluationReport evaluateAgainstAllSolutions(
            NQueensProblem problem,
            RecursiveReasoningRolloutResult rollout,
            NQueensStateTokenDecoder decoder) {
        Objects.requireNonNull(problem, "problem must not be null");
        return evaluate(problem, rollout, decoder, NQueensSolver.count(problem));
    }

    public static NQueensRolloutEvaluationReport evaluate(
            NQueensProblem problem,
            RecursiveReasoningRolloutResult rollout,
            NQueensStateTokenDecoder decoder,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        Objects.requireNonNull(rollout, "rollout must not be null");
        Objects.requireNonNull(decoder, "decoder must not be null");
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        DiscreteRolloutEvaluationReport discreteReport = DiscreteRolloutEvaluator.evaluate(
                rollout,
                state -> decoder.decodeTokens(state, problem),
                prediction -> NQueensBenchmark.evaluateTokensAsDiscrete(problem, prediction.tokens()),
                knownSolutionCount);
        List<NQueensTrajectoryEvaluation> evaluations = new ArrayList<>();
        for (DiscreteTrajectoryEvaluation trajectoryEvaluation : discreteReport.evaluations()) {
            int[] tokens = trajectoryEvaluation.tokens();
            NQueensTokenDecodeResult decoded = NQueensTokenCodec.decodeSolution(problem.size(), tokens);
            NQueensEvaluation evaluation = NQueensBenchmark.evaluateTokens(problem, tokens);

            evaluations.add(new NQueensTrajectoryEvaluation(
                    trajectoryEvaluation.trajectoryIndex(),
                    trajectoryEvaluation.sampleIndex(),
                    trajectoryEvaluation.stateId(),
                    tokens,
                    decoded,
                    evaluation,
                    trajectoryEvaluation.metadata()));
        }

        return new NQueensRolloutEvaluationReport(
                problem,
                evaluations,
                discreteReport.selectedTrajectoryIndex(),
                NQueensBenchmark.toNQueensCoverageReport(discreteReport.coverage()),
                discreteReport.metadata());
    }
}
