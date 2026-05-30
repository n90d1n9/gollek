package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class NQueensSolverTest {
    @Test
    void enumeratesAllFourQueensSolutions() {
        List<NQueensSolution> solutions = NQueensSolver.solve(NQueensProblem.empty(4));

        assertEquals(2, solutions.size());
        assertTrue(solutions.contains(NQueensSolution.ofColumns(1, 3, 0, 2)));
        assertTrue(solutions.contains(NQueensSolution.ofColumns(2, 0, 3, 1)));
        assertEquals(2, NQueensSolver.count(NQueensProblem.empty(4)));
    }

    @Test
    void respectsFixedQueensWhenSolvingPartialProblem() {
        NQueensProblem problem = NQueensProblem.ofFixedColumns(-1, 3, -1, -1);

        List<NQueensSolution> solutions = NQueensSolver.solve(problem);

        assertEquals(1, solutions.size());
        assertEquals(NQueensSolution.ofColumns(1, 3, 0, 2), solutions.getFirst());
    }

    @Test
    void honorsSolutionLimitWithoutCollectingMoreThanNeeded() {
        assertEquals(1, NQueensSolver.solve(NQueensProblem.empty(4), 1).size());
        assertEquals(1, NQueensSolver.count(NQueensProblem.empty(4), 1));
        assertEquals(0, NQueensSolver.solve(NQueensProblem.empty(4), 0).size());
    }

    @Test
    void reportsZeroSolutionsForUnsolvableBoardSize() {
        assertEquals(0, NQueensSolver.count(NQueensProblem.empty(3)));
        assertTrue(NQueensSolver.solve(NQueensProblem.empty(3)).isEmpty());
    }

    @Test
    void coverageAgainstAllSolutionsUsesSolverDenominator() {
        NQueensCoverageReport report = NQueensBenchmark.coverageAgainstAllSolutions(
                NQueensProblem.empty(4),
                List.of(
                        NQueensSolution.ofColumns(1, 3, 0, 2),
                        NQueensSolution.ofColumns(1, 3, 0, 2),
                        NQueensSolution.ofColumns(0, 1, 2, 3)));

        assertEquals(2, report.knownSolutionCount());
        assertEquals(1, report.uniqueValidSolutionCount());
        assertEquals(0.5, report.coverageRate(), 1e-9);
        assertEquals(2.0 / 3.0, report.validRate(), 1e-9);
    }

    @Test
    void rejectsNegativeLimits() {
        assertThrows(IllegalArgumentException.class, () -> NQueensSolver.solve(NQueensProblem.empty(4), -1));
        assertThrows(IllegalArgumentException.class, () -> NQueensSolver.count(NQueensProblem.empty(4), -1));
    }
}
