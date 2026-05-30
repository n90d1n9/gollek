package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetExampleTest {
    @Test
    void storesGenericTokenPairsWithDefensiveCopies() {
        int[] input = {1, 0, 0};
        int[] target = {1, 2, 3};
        DiscreteTokenDatasetExample example = new DiscreteTokenDatasetExample(
                "graph-coloring",
                2,
                input,
                target,
                6,
                Map.of("source", "test"));

        input[0] = 9;
        target[1] = 9;
        int[] exposedInput = example.inputTokens();
        int[] exposedTarget = example.targetTokens();
        exposedInput[1] = 9;
        exposedTarget[2] = 9;

        assertEquals("graph-coloring", example.taskId());
        assertEquals(2, example.exampleIndex());
        assertEquals(3, example.inputTokenCount());
        assertEquals(3, example.targetTokenCount());
        assertTrue(example.hasKnownSolutionCount());
        assertArrayEquals(new int[] {1, 0, 0}, example.inputTokens());
        assertArrayEquals(new int[] {1, 2, 3}, example.targetTokens());
        assertEquals("test", example.metadata().get("source"));
    }

    @Test
    void allowsUnknownSolutionCountButRejectsInvalidDatasetShapes() {
        DiscreteTokenDatasetExample unknown = new DiscreteTokenDatasetExample(
                "arc-grid",
                0,
                new int[] {1},
                new int[] {2},
                -1,
                Map.of());

        assertFalse(unknown.hasKnownSolutionCount());
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetExample("", 0, new int[] {1}, new int[] {2}, 1, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetExample("bad", -1, new int[] {1}, new int[] {2}, 1, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetExample("bad", 0, new int[] {}, new int[] {2}, 1, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetExample("bad", 0, new int[] {1}, new int[] {}, 1, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetExample("bad", 0, new int[] {1}, new int[] {2}, 0, Map.of()));
    }

    @Test
    void nQueensExampleAdaptsToGenericTokenExample() {
        NQueensDatasetExample example = new NQueensDatasetExample(
                3,
                NQueensProblem.ofFixedColumns(-1, 3, -1, -1),
                NQueensSolution.ofColumns(1, 3, 0, 2),
                1,
                Map.of("source", "nq"));

        DiscreteTokenDatasetExample tokenExample = example.toTokenExample();

        assertEquals("nqueens", tokenExample.taskId());
        assertEquals(3, tokenExample.exampleIndex());
        assertEquals(16, tokenExample.inputTokenCount());
        assertEquals(16, tokenExample.targetTokenCount());
        assertArrayEquals(example.inputTokens(), tokenExample.inputTokens());
        assertArrayEquals(example.targetTokens(), tokenExample.targetTokens());
        assertEquals("nq", tokenExample.metadata().get("source"));
        assertEquals(4, tokenExample.metadata().get("size"));
        assertEquals(1, tokenExample.metadata().get("fixedQueenCount"));
        assertEquals(3, tokenExample.metadata().get("removedQueenCount"));
    }

    @Test
    void graphColoringExampleAdaptsToGenericTokenExample() {
        GraphColoringProblem problem = GraphColoringProblem.ofFixedColors(
                3,
                java.util.List.of(GraphColoringEdge.of(0, 1), GraphColoringEdge.of(1, 2)),
                0,
                -1,
                -1);
        GraphColoringDatasetExample example = new GraphColoringDatasetExample(
                4,
                problem,
                GraphColoringSolution.ofColors(3, 0, 1, 0),
                2,
                Map.of("source", "gc"));

        DiscreteTokenDatasetExample tokenExample = example.toTokenExample();

        assertEquals("graph-coloring", tokenExample.taskId());
        assertEquals(4, tokenExample.exampleIndex());
        assertEquals(3, tokenExample.inputTokenCount());
        assertEquals(3, tokenExample.targetTokenCount());
        assertArrayEquals(example.inputTokens(), tokenExample.inputTokens());
        assertArrayEquals(example.targetTokens(), tokenExample.targetTokens());
        assertEquals("gc", tokenExample.metadata().get("source"));
        assertEquals(3, tokenExample.metadata().get("nodeCount"));
        assertEquals(3, tokenExample.metadata().get("colorCount"));
        assertEquals(2, tokenExample.metadata().get("edgeCount"));
        assertEquals(1, tokenExample.metadata().get("fixedColorCount"));
        assertEquals(2, tokenExample.metadata().get("removedColorCount"));
    }
}
