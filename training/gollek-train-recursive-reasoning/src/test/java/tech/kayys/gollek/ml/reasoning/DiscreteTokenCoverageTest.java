package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenCoverageTest {
    @Test
    void summarizesValidUniqueDuplicateAndInvalidCandidates() {
        DiscreteTokenCoverageReport report = DiscreteTokenCoverage.summarize(
                List.of(
                        DiscreteTokenEvaluation.valid("solution-a"),
                        DiscreteTokenEvaluation.valid("solution-b", Map.of("rank", 2)),
                        DiscreteTokenEvaluation.valid("solution-a"),
                        DiscreteTokenEvaluation.invalid(3)),
                4);

        assertEquals(4, report.candidateCount());
        assertEquals(3, report.validCandidateCount());
        assertEquals(2, report.uniqueValidCandidateCount());
        assertEquals(1, report.duplicateValidCandidateCount());
        assertEquals(0.75, report.validRate(), 1e-9);
        assertEquals(0.5, report.coverageRate(), 1e-9);
    }

    @Test
    void emptyCoverageKeepsValidRateZeroAndCoverageUnknown() {
        DiscreteTokenCoverageReport report = DiscreteTokenCoverage.summarize(List.of());

        assertEquals(0, report.candidateCount());
        assertEquals(0.0, report.validRate(), 1e-9);
        assertTrue(Double.isNaN(report.coverageRate()));
        assertEquals(-1, report.knownSolutionCount());
    }

    @Test
    void rejectsInvalidEvaluationAndCoverageInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenEvaluation(true, null, 0, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenEvaluation(true, " ", 0, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenEvaluation.invalid(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenCoverage.summarize(List.of(), -2));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenCoverage.summarize(java.util.Arrays.asList(DiscreteTokenEvaluation.valid("a"), null)));
    }
}
