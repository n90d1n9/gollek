package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class GraphColoringSolverTest {
    @Test
    void enumeratesAllTriangleThreeColorSolutions() {
        GraphColoringProblem problem = triangle(3);

        List<GraphColoringSolution> solutions = GraphColoringSolver.solve(problem);

        assertEquals(6, solutions.size());
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(3, 0, 1, 2)));
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(3, 2, 1, 0)));
        assertEquals(6, GraphColoringSolver.count(problem));
    }

    @Test
    void reportsZeroSolutionsWhenColorCountIsTooSmall() {
        GraphColoringProblem problem = triangle(2);

        assertEquals(0, GraphColoringSolver.count(problem));
        assertTrue(GraphColoringSolver.solve(problem).isEmpty());
    }

    @Test
    void respectsFixedColorsWhenSolvingPartialProblem() {
        GraphColoringProblem problem = GraphColoringProblem.ofFixedColors(
                3,
                List.of(
                        GraphColoringEdge.of(0, 1),
                        GraphColoringEdge.of(1, 2),
                        GraphColoringEdge.of(0, 2)),
                0,
                -1,
                -1);

        List<GraphColoringSolution> solutions = GraphColoringSolver.solve(problem);

        assertEquals(2, solutions.size());
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(3, 0, 1, 2)));
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(3, 0, 2, 1)));
        assertEquals(2, GraphColoringSolver.count(problem));
    }

    @Test
    void handlesPathGraphWithTwoColors() {
        GraphColoringProblem problem = GraphColoringProblem.ofEdges(
                3,
                2,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2));

        List<GraphColoringSolution> solutions = GraphColoringSolver.solve(problem);

        assertEquals(2, solutions.size());
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(2, 0, 1, 0)));
        assertTrue(solutions.contains(GraphColoringSolution.ofColors(2, 1, 0, 1)));
    }

    @Test
    void honorsSolutionLimitWithoutCollectingMoreThanNeeded() {
        GraphColoringProblem problem = triangle(3);

        assertEquals(2, GraphColoringSolver.solve(problem, 2).size());
        assertEquals(2, GraphColoringSolver.count(problem, 2));
        assertEquals(0, GraphColoringSolver.solve(problem, 0).size());
    }

    @Test
    void coverageAgainstAllSolutionsUsesSolverDenominator() {
        GraphColoringProblem problem = triangle(3);
        GraphColoringCoverageReport report = GraphColoringBenchmark.coverageAgainstAllSolutions(
                problem,
                List.of(
                        GraphColoringSolution.ofColors(3, 0, 1, 2),
                        GraphColoringSolution.ofColors(3, 0, 1, 2),
                        GraphColoringSolution.ofColors(3, 0, 0, 1)));

        assertEquals(6, report.knownSolutionCount());
        assertEquals(1, report.uniqueValidSolutionCount());
        assertEquals(1.0 / 6.0, report.coverageRate(), 1e-9);
        assertEquals(2.0 / 3.0, report.validRate(), 1e-9);
    }

    @Test
    void tokenCoverageAgainstAllSolutionsUsesSolverDenominator() {
        GraphColoringProblem problem = triangle(3);
        GraphColoringCoverageReport report = GraphColoringBenchmark.coverageTokensAgainstAllSolutions(
                problem,
                List.of(
                        GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens(),
                        GraphColoringSolution.ofColors(3, 0, 2, 1).toTokens(),
                        new int[] {1, 1, 0}));

        assertEquals(6, report.knownSolutionCount());
        assertEquals(2, report.uniqueValidSolutionCount());
        assertEquals(2.0 / 6.0, report.coverageRate(), 1e-9);
        assertEquals(2.0 / 3.0, report.validRate(), 1e-9);
    }

    @Test
    void rejectsNegativeLimits() {
        GraphColoringProblem problem = triangle(3);

        assertThrows(IllegalArgumentException.class, () -> GraphColoringSolver.solve(problem, -1));
        assertThrows(IllegalArgumentException.class, () -> GraphColoringSolver.count(problem, -1));
    }

    private static GraphColoringProblem triangle(int colorCount) {
        return GraphColoringProblem.ofEdges(
                3,
                colorCount,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2),
                GraphColoringEdge.of(0, 2));
    }
}
