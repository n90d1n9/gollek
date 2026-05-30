package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetBatcherTest {
    @Test
    void padsExamplesIntoBackendNeutralBatch() {
        DiscreteTokenDatasetBatch batch = DiscreteTokenDatasetBatcher.batch(
                List.of(
                        new DiscreteTokenDatasetExample(
                                "graph-coloring",
                                2,
                                new int[] {1, 0, 0},
                                new int[] {1, 2, 3},
                                6,
                                Map.of("source", "gc")),
                        new DiscreteTokenDatasetExample(
                                "nqueens",
                                3,
                                new int[] {1, 2},
                                new int[] {2},
                                -1,
                                Map.of("source", "nq"))),
                -10,
                -20);

        assertEquals(2, batch.batchSize());
        assertEquals(3, batch.maxInputLength());
        assertEquals(3, batch.maxTargetLength());
        assertArrayEquals(new String[] {"graph-coloring", "nqueens"}, batch.taskIds());
        assertArrayEquals(new int[] {2, 3}, batch.exampleIndices());
        assertArrayEquals(new int[] {3, 2}, batch.inputLengths());
        assertArrayEquals(new int[] {3, 1}, batch.targetLengths());
        assertArrayEquals(new int[] {6, -1}, batch.knownSolutionCounts());
        assertArrayEquals(new int[] {1, 0, 0}, batch.inputTokens()[0]);
        assertArrayEquals(new int[] {1, 2, -10}, batch.inputTokens()[1]);
        assertArrayEquals(new int[] {1, 2, 3}, batch.targetTokens()[0]);
        assertArrayEquals(new int[] {2, -20, -20}, batch.targetTokens()[1]);
        assertArrayEquals(new int[] {1, 1, 1}, batch.inputMask()[0]);
        assertArrayEquals(new int[] {1, 1, 0}, batch.inputMask()[1]);
        assertArrayEquals(new int[] {1, 1, 1}, batch.targetMask()[0]);
        assertArrayEquals(new int[] {1, 0, 0}, batch.targetMask()[1]);
        assertEquals("gc", batch.metadata().get(0).get("source"));
        assertEquals("nq", batch.metadata().get(1).get("source"));
        assertEquals(3, batch.inputTokenCount(0));
        assertEquals(1, batch.targetTokenCount(1));
        assertTrue(batch.hasKnownSolutionCount(0));
        assertFalse(batch.hasKnownSolutionCount(1));
        assertEquals(-10, batch.inputPadToken());
        assertEquals(-20, batch.targetPadToken());
    }

    @Test
    void exposesDefensiveCopies() {
        DiscreteTokenDatasetBatch batch = DiscreteTokenDatasetBatcher.batch(
                List.of(new DiscreteTokenDatasetExample(
                        "graph-coloring",
                        2,
                        new int[] {1, 0},
                        new int[] {1},
                        6,
                        Map.of("source", "gc"))),
                -1);

        batch.taskIds()[0] = "changed";
        batch.exampleIndices()[0] = 99;
        batch.inputTokens()[0][0] = 99;
        batch.targetTokens()[0][0] = 99;
        batch.inputMask()[0][0] = 0;
        batch.targetMask()[0][0] = 0;
        batch.inputLengths()[0] = 99;
        batch.targetLengths()[0] = 99;
        batch.knownSolutionCounts()[0] = 99;

        assertArrayEquals(new String[] {"graph-coloring"}, batch.taskIds());
        assertArrayEquals(new int[] {2}, batch.exampleIndices());
        assertArrayEquals(new int[] {1, 0}, batch.inputTokens()[0]);
        assertArrayEquals(new int[] {1}, batch.targetTokens()[0]);
        assertArrayEquals(new int[] {1, 1}, batch.inputMask()[0]);
        assertArrayEquals(new int[] {1}, batch.targetMask()[0]);
        assertArrayEquals(new int[] {2}, batch.inputLengths());
        assertArrayEquals(new int[] {1}, batch.targetLengths());
        assertArrayEquals(new int[] {6}, batch.knownSolutionCounts());
    }

    @Test
    void rejectsMalformedBatches() {
        assertThrows(IllegalArgumentException.class, () -> DiscreteTokenDatasetBatcher.batch(List.of(), -1));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetBatcher.batch(List.of((DiscreteTokenDatasetExample) null), -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetBatch(
                        new String[] {"bad"},
                        new int[] {0},
                        new int[][] {{1, 2}},
                        new int[][] {{1}},
                        new int[][] {{1, 0}},
                        new int[][] {{1}},
                        new int[] {2},
                        new int[] {1},
                        new int[] {0},
                        List.of(Map.of()),
                        -1,
                        -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetBatch(
                        new String[] {"bad"},
                        new int[] {0},
                        new int[][] {{1, 2}},
                        new int[][] {{1}},
                        new int[][] {{1, 1}},
                        new int[][] {{1}},
                        new int[] {1},
                        new int[] {1},
                        new int[] {1},
                        List.of(Map.of()),
                        -1,
                        -1));
    }
}
