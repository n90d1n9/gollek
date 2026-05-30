package tech.kayys.gollek.ml.reasoning;

/**
 * Multi-sample coverage report for benchmark-neutral discrete token candidates.
 */
public record DiscreteTokenCoverageReport(
        int candidateCount,
        int validCandidateCount,
        int uniqueValidCandidateCount,
        int duplicateValidCandidateCount,
        int knownSolutionCount,
        double validRate,
        double coverageRate) {

    public DiscreteTokenCoverageReport {
        candidateCount = requireNonNegative(candidateCount, "candidateCount");
        validCandidateCount = requireNonNegative(validCandidateCount, "validCandidateCount");
        uniqueValidCandidateCount = requireNonNegative(uniqueValidCandidateCount, "uniqueValidCandidateCount");
        duplicateValidCandidateCount = requireNonNegative(
                duplicateValidCandidateCount,
                "duplicateValidCandidateCount");
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }
        if (validCandidateCount > candidateCount) {
            throw new IllegalArgumentException("validCandidateCount cannot exceed candidateCount");
        }
        if (uniqueValidCandidateCount > validCandidateCount) {
            throw new IllegalArgumentException("uniqueValidCandidateCount cannot exceed validCandidateCount");
        }
        if (!Double.isFinite(validRate) || validRate < 0.0 || validRate > 1.0) {
            throw new IllegalArgumentException("validRate must be finite and in [0, 1] but was " + validRate);
        }
        if (!(Double.isNaN(coverageRate)
                || Double.isFinite(coverageRate) && coverageRate >= 0.0 && coverageRate <= 1.0)) {
            throw new IllegalArgumentException(
                    "coverageRate must be NaN or finite and in [0, 1] but was " + coverageRate);
        }
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }
}
