package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class NQueensBenchmarkTest {
    @Test
    void evaluatesValidFourQueensSolution() {
        NQueensProblem problem = NQueensProblem.empty(4);
        NQueensSolution solution = NQueensSolution.ofColumns(1, 3, 0, 2);

        NQueensEvaluation evaluation = NQueensBenchmark.evaluate(problem, solution);

        assertTrue(evaluation.valid());
        assertEquals(0, evaluation.conflictCount());
        assertEquals(0, evaluation.diagonalConflictCount());
        assertEquals(0, evaluation.columnConflictCount());
        assertEquals(4, evaluation.metadata().get("size"));
    }

    @Test
    void reportsFixedQueenAndDiagonalConflicts() {
        NQueensProblem problem = NQueensProblem.ofFixedColumns(1, -1, -1, -1);
        NQueensSolution solution = NQueensSolution.ofColumns(0, 1, 2, 3);

        NQueensEvaluation evaluation = NQueensBenchmark.evaluate(problem, solution);

        assertFalse(evaluation.valid());
        assertEquals(1, evaluation.fixedViolationCount());
        assertEquals(6, evaluation.diagonalConflictCount());
        assertEquals(7, evaluation.conflictCount());
    }

    @Test
    void coverageCountsUniqueValidSolutionsAndDuplicates() {
        NQueensProblem problem = NQueensProblem.empty(4);
        NQueensCoverageReport report = NQueensBenchmark.coverage(
                problem,
                List.of(
                        NQueensSolution.ofColumns(1, 3, 0, 2),
                        NQueensSolution.ofColumns(2, 0, 3, 1),
                        NQueensSolution.ofColumns(1, 3, 0, 2),
                        NQueensSolution.ofColumns(0, 1, 2, 3)),
                2);

        assertEquals(4, report.candidateCount());
        assertEquals(3, report.validCandidateCount());
        assertEquals(2, report.uniqueValidSolutionCount());
        assertEquals(1, report.duplicateValidCandidateCount());
        assertEquals(0.75, report.validRate(), 1e-9);
        assertEquals(1.0, report.coverageRate(), 1e-9);
    }

    @Test
    void tokenEncodingUsesPaperStyleEmptyAndQueenTokens() {
        NQueensProblem problem = NQueensProblem.ofFixedColumns(-1, 2, -1, -1);
        NQueensSolution solution = NQueensSolution.ofColumns(1, 3, 0, 2);

        assertArrayEquals(
                new int[] {
                    1, 1, 1, 1,
                    1, 1, 2, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1
                },
                problem.toTokens());
        assertArrayEquals(
                new int[] {
                    1, 2, 1, 1,
                    1, 1, 1, 2,
                    2, 1, 1, 1,
                    1, 1, 2, 1
                },
                solution.toTokens());
    }

    @Test
    void problemAndSolutionDefensivelyCopyColumns() {
        int[] fixed = {-1, 3, -1, -1};
        int[] candidate = {1, 3, 0, 2};
        NQueensProblem problem = new NQueensProblem(4, fixed);
        NQueensSolution solution = new NQueensSolution(candidate);

        fixed[1] = 0;
        candidate[0] = 0;
        problem.fixedColumnsByRow()[1] = 0;
        solution.columnsByRow()[0] = 0;

        assertEquals(3, problem.fixedColumn(1));
        assertEquals(1, solution.column(0));
    }

    @Test
    void rejectsImpossibleFixedQueensAndMismatchedCandidateSize() {
        assertThrows(IllegalArgumentException.class, () -> NQueensProblem.ofFixedColumns(0, 0, -1, -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> NQueensBenchmark.evaluate(NQueensProblem.empty(4), NQueensSolution.ofColumns(1, 3, 0)));
    }
}
