package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Composes rollout token collection, candidate scoring, and coverage aggregation.
 */
public final class DiscreteRolloutEvaluator {
    private DiscreteRolloutEvaluator() {
    }

    public static DiscreteRolloutEvaluationReport evaluate(
            RecursiveReasoningRolloutResult rollout,
            DiscreteStateTokenDecoder decoder,
            DiscreteTokenCandidateEvaluator candidateEvaluator) {
        return evaluate(rollout, decoder, candidateEvaluator, -1);
    }

    public static DiscreteRolloutEvaluationReport evaluate(
            RecursiveReasoningRolloutResult rollout,
            DiscreteStateTokenDecoder decoder,
            DiscreteTokenCandidateEvaluator candidateEvaluator,
            int knownSolutionCount) {
        DiscreteRolloutTokenReport tokenReport = DiscreteRolloutTokenCollector.collectFinalStateTokens(
                rollout,
                decoder);
        return evaluate(tokenReport, candidateEvaluator, knownSolutionCount);
    }

    public static DiscreteRolloutEvaluationReport evaluate(
            DiscreteRolloutTokenReport tokenReport,
            DiscreteTokenCandidateEvaluator candidateEvaluator) {
        return evaluate(tokenReport, candidateEvaluator, -1);
    }

    public static DiscreteRolloutEvaluationReport evaluate(
            DiscreteRolloutTokenReport tokenReport,
            DiscreteTokenCandidateEvaluator candidateEvaluator,
            int knownSolutionCount) {
        Objects.requireNonNull(tokenReport, "tokenReport must not be null");
        Objects.requireNonNull(candidateEvaluator, "candidateEvaluator must not be null");
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        List<DiscreteTrajectoryEvaluation> evaluations = new ArrayList<>();
        List<DiscreteTokenEvaluation> candidateEvaluations = new ArrayList<>();
        for (DiscreteTrajectoryTokenPrediction prediction : tokenReport.predictions()) {
            DiscreteTokenEvaluation evaluation = Objects.requireNonNull(
                    candidateEvaluator.evaluate(prediction),
                    "candidateEvaluator returned null for state " + prediction.stateId());
            candidateEvaluations.add(evaluation);
            evaluations.add(new DiscreteTrajectoryEvaluation(
                    prediction.trajectoryIndex(),
                    prediction.sampleIndex(),
                    prediction.stateId(),
                    prediction.tokens(),
                    evaluation,
                    prediction.metadata()));
        }

        DiscreteTokenCoverageReport coverage = DiscreteTokenCoverage.summarize(
                candidateEvaluations,
                knownSolutionCount);
        return new DiscreteRolloutEvaluationReport(
                evaluations,
                tokenReport.selectedTrajectoryIndex(),
                coverage,
                reportMetadata(tokenReport, knownSolutionCount));
    }

    private static Map<String, Object> reportMetadata(
            DiscreteRolloutTokenReport tokenReport,
            int knownSolutionCount) {
        Map<String, Object> metadata = new LinkedHashMap<>(tokenReport.metadata());
        metadata.put("knownSolutionCount", knownSolutionCount);
        return metadata;
    }
}
