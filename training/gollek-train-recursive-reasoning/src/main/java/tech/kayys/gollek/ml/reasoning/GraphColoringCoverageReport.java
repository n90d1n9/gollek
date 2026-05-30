package tech.kayys.gollek.ml.reasoning;

/**
 * Multi-sample graph-coloring coverage report.
 */
public record GraphColoringCoverageReport(
        int candidateCount,
        int validCandidateCount,
        int uniqueValidSolutionCount,
        int duplicateValidCandidateCount,
        int knownSolutionCount,
        double validRate,
        double coverageRate) {

    public GraphColoringCoverageReport {
        candidateCount = requireNonNegative(candidateCount, "candidateCount");
        validCandidateCount = requireNonNegative(validCandidateCount, "validCandidateCount");
        uniqueValidSolutionCount = requireNonNegative(uniqueValidSolutionCount, "uniqueValidSolutionCount");
        duplicateValidCandidateCount = requireNonNegative(
                duplicateValidCandidateCount,
                "duplicateValidCandidateCount");
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }
        if (validCandidateCount > candidateCount) {
            throw new IllegalArgumentException("validCandidateCount cannot exceed candidateCount");
        }
        if (uniqueValidSolutionCount > validCandidateCount) {
            throw new IllegalArgumentException("uniqueValidSolutionCount cannot exceed validCandidateCount");
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
