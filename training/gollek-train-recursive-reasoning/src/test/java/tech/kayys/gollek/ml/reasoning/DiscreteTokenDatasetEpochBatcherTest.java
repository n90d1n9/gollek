package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetEpochBatcherTest {
    @Test
    void buildsSequentialMiniBatchesAndKeepsPartialTail() {
        DiscreteTokenDatasetEpoch epoch =
                DiscreteTokenDatasetEpochBatcher.sequential(examples(), 2, -10, -20, false);

        assertEquals(3, epoch.exampleCount());
        assertEquals(3, epoch.emittedExampleCount());
        assertEquals(0, epoch.droppedExampleCount());
        assertEquals(2, epoch.requestedBatchSize());
        assertFalse(epoch.shuffled());
        assertFalse(epoch.dropLast());
        assertEquals(0L, epoch.seed());
        assertEquals(-10, epoch.inputPadToken());
        assertEquals(-20, epoch.targetPadToken());
        assertEquals(2, epoch.batchCount());
        assertTrue(epoch.hasBatches());
        assertTrue(epoch.emittedAllExamples());

        assertArrayEquals(new int[] {0, 1}, epoch.batches().get(0).exampleIndices());
        assertArrayEquals(new int[] {2}, epoch.batches().get(1).exampleIndices());
        assertEquals(1, epoch.batches().get(1).batchSize());
    }

    @Test
    void dropsPartialTailWhenRequested() {
        DiscreteTokenDatasetEpoch epoch =
                DiscreteTokenDatasetEpochBatcher.sequential(examples(), 2, -1, -1, true);

        assertEquals(3, epoch.exampleCount());
        assertEquals(2, epoch.emittedExampleCount());
        assertEquals(1, epoch.droppedExampleCount());
        assertEquals(1, epoch.batchCount());
        assertFalse(epoch.emittedAllExamples());
        assertArrayEquals(new int[] {0, 1}, epoch.batches().get(0).exampleIndices());
    }

    @Test
    void shuffledMiniBatchesAreDeterministicBySeed() {
        DiscreteTokenDatasetEpoch first = DiscreteTokenDatasetEpochBatcher.shuffled(examples(), 2, -1, 123L, false);
        DiscreteTokenDatasetEpoch second = DiscreteTokenDatasetEpochBatcher.shuffled(examples(), 2, -1, 123L, false);

        assertTrue(first.shuffled());
        assertEquals(123L, first.seed());
        assertArrayEquals(
                first.batches().get(0).exampleIndices(),
                second.batches().get(0).exampleIndices());
        assertArrayEquals(
                first.batches().get(1).exampleIndices(),
                second.batches().get(1).exampleIndices());
    }

    @Test
    void lengthSortedMiniBatchesReducePaddingWaste() {
        List<DiscreteTokenDatasetExample> varied = List.of(
                variableLengthExample(0, 1),
                variableLengthExample(1, 8),
                variableLengthExample(2, 2),
                variableLengthExample(3, 7));

        DiscreteTokenDatasetEpoch sequential =
                DiscreteTokenDatasetEpochBatcher.sequential(varied, 2, -1);
        DiscreteTokenDatasetEpoch lengthSorted =
                DiscreteTokenDatasetEpochBatcher.lengthSorted(varied, 2, -1, false);

        assertArrayEquals(new int[] {0, 2}, lengthSorted.batches().get(0).exampleIndices());
        assertArrayEquals(new int[] {3, 1}, lengthSorted.batches().get(1).exampleIndices());
        assertEquals(22L, lengthSorted.observedTokenCount());
        assertEquals(18L, lengthSorted.inputTokenCount());
        assertEquals(4L, lengthSorted.targetTokenCount());
        assertEquals(24L, lengthSorted.paddedTokenCapacity());
        assertEquals(2L, lengthSorted.inputPaddingTokenCount());
        assertEquals(0L, lengthSorted.targetPaddingTokenCount());
        assertEquals(2L, lengthSorted.paddingTokenCount());
        assertEquals(2.0d / 24.0d, lengthSorted.paddingRate(), 1e-9);
        assertTrue(lengthSorted.paddingTokenCount() < sequential.paddingTokenCount());
        assertFalse(lengthSorted.shuffled());
        assertFalse(lengthSorted.dropLast());
    }

    @Test
    void handlesEmptyDatasetsAsEmptyEpochs() {
        DiscreteTokenDatasetEpoch epoch = DiscreteTokenDatasetEpochBatcher.sequential(List.of(), 4, -1);

        assertEquals(0, epoch.exampleCount());
        assertEquals(0, epoch.emittedExampleCount());
        assertEquals(0, epoch.droppedExampleCount());
        assertEquals(0, epoch.batchCount());
        assertFalse(epoch.hasBatches());
        assertTrue(epoch.emittedAllExamples());
        assertEquals(0L, epoch.observedTokenCount());
        assertEquals(0L, epoch.paddedTokenCapacity());
        assertEquals(0L, epoch.paddingTokenCount());
        assertEquals(0.0d, epoch.paddingRate(), 1e-9);
    }

    @Test
    void rejectsInvalidEpochInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetEpochBatcher.sequential(examples(), 0, -1));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetEpochBatcher.sequential(
                        List.of(example(0), null),
                        2,
                        -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetEpoch(
                        List.of(DiscreteTokenDatasetBatcher.batch(List.of(example(0)), -1)),
                        2,
                        1,
                        0,
                        1,
                        false,
                        false,
                        0L,
                        -1,
                        -1));
    }

    private static List<DiscreteTokenDatasetExample> examples() {
        return List.of(
                example(0),
                new DiscreteTokenDatasetExample(
                        "nqueens",
                        1,
                        new int[] {3},
                        new int[] {4, 5},
                        -1,
                        Map.of("source", "b")),
                new DiscreteTokenDatasetExample(
                        "arc-grid",
                        2,
                        new int[] {6, 7, 8},
                        new int[] {9},
                        2,
                        Map.of("source", "c")));
    }

    private static DiscreteTokenDatasetExample example(int index) {
        return new DiscreteTokenDatasetExample(
                "graph-coloring",
                index,
                new int[] {1, 2},
                new int[] {2},
                1,
                Map.of("source", "a"));
    }

    private static DiscreteTokenDatasetExample variableLengthExample(int index, int inputLength) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                "variable",
                index,
                input,
                new int[] {index + 100},
                1,
                Map.of("source", "variable"));
    }
}
