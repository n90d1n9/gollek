package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class NQueensDatasetGeneratorTest {
    @Test
    void generatesDeterministicPartialBoardExamples() {
        List<NQueensDatasetExample> first = NQueensDatasetGenerator.generate(4, 4, 2, 17L);
        List<NQueensDatasetExample> second = NQueensDatasetGenerator.generate(4, 4, 2, 17L);

        assertEquals(4, first.size());
        assertEquals(4, second.size());
        for (int i = 0; i < first.size(); i++) {
            NQueensDatasetExample example = first.get(i);
            assertEquals(i, example.exampleIndex());
            assertEquals(2, example.fixedQueenCount());
            assertEquals(2, example.removedQueenCount());
            assertEquals(16, example.inputTokens().length);
            assertEquals(16, example.targetTokens().length);
            assertTrue(example.knownSolutionCount() >= 1);
            assertTrue(NQueensBenchmark.evaluate(example.problem(), example.targetSolution()).valid());
            assertArrayEquals(example.inputTokens(), second.get(i).inputTokens());
            assertArrayEquals(example.targetTokens(), second.get(i).targetTokens());
        }
    }

    @Test
    void preservesSourceFixedQueensWhenGeneratingFromPartialProblem() {
        NQueensProblem source = NQueensProblem.ofFixedColumns(-1, 3, -1, -1);

        List<NQueensDatasetExample> examples = NQueensDatasetGenerator.generate(source, 3, 2, 9L);

        assertEquals(3, examples.size());
        for (NQueensDatasetExample example : examples) {
            assertEquals(3, example.problem().fixedColumn(1));
            assertEquals(2, example.fixedQueenCount());
            assertEquals(NQueensSolution.ofColumns(1, 3, 0, 2), example.targetSolution());
            assertEquals(1, example.metadata().get("sourceFixedQueenCount"));
            assertEquals(1, example.metadata().get("sourceSolutionCount"));
        }
    }

    @Test
    void zeroExamplesDoesNotRequireSolvableSourceProblem() {
        assertTrue(NQueensDatasetGenerator.generate(NQueensProblem.empty(3), 0, 0, 1L).isEmpty());
    }

    @Test
    void rejectsInvalidRequestsAndUnsolvableSourceProblems() {
        assertThrows(IllegalArgumentException.class, () -> NQueensDatasetGenerator.generate(4, -1, 2, 1L));
        assertThrows(IllegalArgumentException.class, () -> NQueensDatasetGenerator.generate(4, 1, 5, 1L));
        assertThrows(
                IllegalArgumentException.class,
                () -> NQueensDatasetGenerator.generate(NQueensProblem.ofFixedColumns(-1, 3, -1, -1), 1, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> NQueensDatasetGenerator.generate(3, 1, 0, 1L));
    }

    @Test
    void problemEqualityUsesColumnContents() {
        assertEquals(
                NQueensProblem.ofFixedColumns(-1, 3, -1, -1),
                NQueensProblem.ofFixedColumns(-1, 3, -1, -1));
        assertEquals(
                NQueensProblem.ofFixedColumns(-1, 3, -1, -1).hashCode(),
                NQueensProblem.ofFixedColumns(-1, 3, -1, -1).hashCode());
    }
}
