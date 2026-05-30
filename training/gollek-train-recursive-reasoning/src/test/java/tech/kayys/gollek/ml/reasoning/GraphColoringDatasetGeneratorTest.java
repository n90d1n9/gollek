package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class GraphColoringDatasetGeneratorTest {
    @Test
    void generatesDeterministicPartialColoringExamples() {
        GraphColoringProblem source = triangle(3);

        List<GraphColoringDatasetExample> first = GraphColoringDatasetGenerator.generate(source, 4, 2, 31L);
        List<GraphColoringDatasetExample> second = GraphColoringDatasetGenerator.generate(source, 4, 2, 31L);

        assertEquals(4, first.size());
        assertEquals(4, second.size());
        for (int i = 0; i < first.size(); i++) {
            GraphColoringDatasetExample example = first.get(i);
            assertEquals(i, example.exampleIndex());
            assertEquals(2, example.fixedColorCount());
            assertEquals(1, example.removedColorCount());
            assertEquals(3, example.inputTokens().length);
            assertEquals(3, example.targetTokens().length);
            assertTrue(example.knownSolutionCount() >= 1);
            assertTrue(GraphColoringBenchmark.evaluate(example.problem(), example.targetSolution()).valid());
            assertArrayEquals(example.inputTokens(), second.get(i).inputTokens());
            assertArrayEquals(example.targetTokens(), second.get(i).targetTokens());
        }
    }

    @Test
    void preservesSourceFixedColorsWhenGeneratingFromPartialProblem() {
        GraphColoringProblem source = GraphColoringProblem.ofFixedColors(
                3,
                List.of(
                        GraphColoringEdge.of(0, 1),
                        GraphColoringEdge.of(1, 2),
                        GraphColoringEdge.of(0, 2)),
                0,
                -1,
                -1);

        List<GraphColoringDatasetExample> examples = GraphColoringDatasetGenerator.generate(source, 3, 2, 11L);

        assertEquals(3, examples.size());
        for (GraphColoringDatasetExample example : examples) {
            assertEquals(0, example.problem().fixedColor(0));
            assertEquals(2, example.fixedColorCount());
            assertTrue(example.targetSolution().equals(GraphColoringSolution.ofColors(3, 0, 1, 2))
                    || example.targetSolution().equals(GraphColoringSolution.ofColors(3, 0, 2, 1)));
            assertEquals(1, example.metadata().get("sourceFixedColorCount"));
            assertEquals(2, example.metadata().get("sourceSolutionCount"));
            assertEquals(1, example.knownSolutionCount());
        }
    }

    @Test
    void zeroExamplesDoesNotRequireSolvableSourceProblem() {
        assertTrue(GraphColoringDatasetGenerator.generate(triangle(2), 0, 0, 1L).isEmpty());
    }

    @Test
    void rejectsInvalidRequestsAndUnsolvableSourceProblems() {
        GraphColoringProblem source = GraphColoringProblem.ofFixedColors(
                3,
                List.of(GraphColoringEdge.of(0, 1)),
                0,
                -1);

        assertThrows(IllegalArgumentException.class, () -> GraphColoringDatasetGenerator.generate(source, -1, 1, 1L));
        assertThrows(IllegalArgumentException.class, () -> GraphColoringDatasetGenerator.generate(source, 1, 3, 1L));
        assertThrows(IllegalArgumentException.class, () -> GraphColoringDatasetGenerator.generate(source, 1, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> GraphColoringDatasetGenerator.generate(triangle(2), 1, 0, 1L));
    }

    @Test
    void exampleRejectsInvalidTargetSolution() {
        GraphColoringProblem source = triangle(3);

        assertThrows(
                IllegalArgumentException.class,
                () -> new GraphColoringDatasetExample(
                        0,
                        source,
                        GraphColoringSolution.ofColors(3, 0, 0, 1),
                        1,
                        java.util.Map.of()));
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
