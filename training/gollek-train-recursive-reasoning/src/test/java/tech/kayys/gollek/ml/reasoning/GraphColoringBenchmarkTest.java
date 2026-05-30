package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class GraphColoringBenchmarkTest {
    @Test
    void evaluatesValidTriangleColoring() {
        GraphColoringProblem problem = triangle();
        GraphColoringSolution solution = GraphColoringSolution.ofColors(3, 0, 1, 2);

        GraphColoringEvaluation evaluation = GraphColoringBenchmark.evaluate(problem, solution);

        assertTrue(evaluation.valid());
        assertEquals(0, evaluation.conflictCount());
        assertEquals(3, evaluation.metadata().get("nodeCount"));
        assertEquals(3, evaluation.metadata().get("edgeCount"));
    }

    @Test
    void reportsFixedColorEdgeAndUncoloredConflicts() {
        GraphColoringProblem problem = GraphColoringProblem.ofFixedColors(
                3,
                List.of(GraphColoringEdge.of(0, 1), GraphColoringEdge.of(1, 2)),
                0,
                -1,
                -1);
        GraphColoringSolution solution = GraphColoringSolution.ofColors(3, 1, 1, -1);

        GraphColoringEvaluation evaluation = GraphColoringBenchmark.evaluate(problem, solution);

        assertFalse(evaluation.valid());
        assertEquals(1, evaluation.fixedViolationCount());
        assertEquals(1, evaluation.edgeConflictCount());
        assertEquals(1, evaluation.uncoloredNodeCount());
        assertEquals(3, evaluation.conflictCount());
    }

    @Test
    void tokenCodecUsesZeroForUncoloredAndOneBasedColors() {
        GraphColoringProblem problem = GraphColoringProblem.ofFixedColors(
                3,
                List.of(GraphColoringEdge.of(0, 1)),
                0,
                -1,
                2);
        GraphColoringSolution solution = GraphColoringSolution.ofColors(3, 0, 1, 2);

        assertArrayEquals(new int[] {1, 0, 3}, GraphColoringTokenCodec.encode(problem));
        assertArrayEquals(new int[] {1, 2, 3}, GraphColoringTokenCodec.encode(solution));
        assertEquals(solution, GraphColoringTokenCodec.decodeSolution(3, 3, solution.toTokens()).solution());
    }

    @Test
    void evaluatesTokenPredictionsWithDecodeDiagnostics() {
        GraphColoringProblem problem = triangle();
        int[] tokens = {1, 1, 9};

        GraphColoringEvaluation evaluation = GraphColoringBenchmark.evaluateTokens(problem, tokens);
        GraphColoringTokenDecodeResult decoded = GraphColoringTokenCodec.decodeSolution(3, 3, tokens);

        assertFalse(evaluation.valid());
        assertEquals(1, evaluation.edgeConflictCount());
        assertEquals(1, evaluation.invalidTokenCount());
        assertEquals(3, evaluation.conflictCount());
        assertEquals(1, decoded.invalidTokenCount());
        assertEquals(1, decoded.uncoloredNodeCount());
    }

    @Test
    void coverageCountsUniqueValidColoringsAndDuplicates() {
        GraphColoringProblem problem = triangle();
        GraphColoringCoverageReport report = GraphColoringBenchmark.coverage(
                problem,
                List.of(
                        GraphColoringSolution.ofColors(3, 0, 1, 2),
                        GraphColoringSolution.ofColors(3, 0, 2, 1),
                        GraphColoringSolution.ofColors(3, 0, 1, 2),
                        GraphColoringSolution.ofColors(3, 0, 0, 1)),
                6);

        assertEquals(4, report.candidateCount());
        assertEquals(3, report.validCandidateCount());
        assertEquals(2, report.uniqueValidSolutionCount());
        assertEquals(1, report.duplicateValidCandidateCount());
        assertEquals(0.75, report.validRate(), 1e-9);
        assertEquals(2.0 / 6.0, report.coverageRate(), 1e-9);
    }

    @Test
    void rejectsInvalidGraphInputs() {
        assertThrows(IllegalArgumentException.class, () -> GraphColoringEdge.of(0, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> GraphColoringProblem.ofEdges(2, 2, GraphColoringEdge.of(0, 2)));
        assertThrows(
                IllegalArgumentException.class,
                () -> GraphColoringProblem.ofEdges(
                        2,
                        2,
                        GraphColoringEdge.of(0, 1),
                        GraphColoringEdge.of(1, 0)));
        assertThrows(
                IllegalArgumentException.class,
                () -> GraphColoringProblem.ofFixedColors(
                        2,
                        List.of(GraphColoringEdge.of(0, 1)),
                        0,
                        0));
        assertThrows(
                IllegalArgumentException.class,
                () -> GraphColoringBenchmark.evaluate(
                        triangle(),
                        GraphColoringSolution.ofColors(2, 0, 1, 0)));
    }

    private static GraphColoringProblem triangle() {
        return GraphColoringProblem.ofEdges(
                3,
                3,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2),
                GraphColoringEdge.of(0, 2));
    }
}
