package tech.kayys.gollek.ml.reasoning;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Shared valid-rate and unique-solution coverage aggregation.
 */
public final class DiscreteTokenCoverage {
    private DiscreteTokenCoverage() {
    }

    public static DiscreteTokenCoverageReport summarize(List<DiscreteTokenEvaluation> evaluations) {
        return summarize(evaluations, -1);
    }

    public static DiscreteTokenCoverageReport summarize(
            List<DiscreteTokenEvaluation> evaluations,
            int knownSolutionCount) {
        List<DiscreteTokenEvaluation> evaluationList = List.copyOf(
                Objects.requireNonNull(evaluations, "evaluations must not be null"));
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        int validCandidates = 0;
        Set<String> uniqueValidCandidates = new HashSet<>();
        for (DiscreteTokenEvaluation evaluation : evaluationList) {
            Objects.requireNonNull(evaluation, "evaluations must not contain null entries");
            if (evaluation.valid()) {
                validCandidates++;
                uniqueValidCandidates.add(evaluation.canonicalKey());
            }
        }

        int duplicateValidCandidates = validCandidates - uniqueValidCandidates.size();
        double validRate = evaluationList.isEmpty() ? 0.0 : (double) validCandidates / evaluationList.size();
        double coverageRate = knownSolutionCount > 0
                ? Math.min(1.0, (double) uniqueValidCandidates.size() / knownSolutionCount)
                : Double.NaN;
        return new DiscreteTokenCoverageReport(
                evaluationList.size(),
                validCandidates,
                uniqueValidCandidates.size(),
                duplicateValidCandidates,
                knownSolutionCount,
                validRate,
                coverageRate);
    }
}
