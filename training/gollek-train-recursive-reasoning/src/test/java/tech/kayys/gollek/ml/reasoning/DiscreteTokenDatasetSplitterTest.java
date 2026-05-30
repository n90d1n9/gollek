package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetSplitterTest {
    @Test
    void splitsSequentiallyByCounts() {
        DiscreteTokenDatasetSplit split = DiscreteTokenDatasetSplitter.sequentialByCounts(examples(5), 1, 2);

        assertEquals(5, split.exampleCount());
        assertEquals(2, split.trainCount());
        assertEquals(1, split.validationCount());
        assertEquals(2, split.testCount());
        assertFalse(split.shuffled());
        assertEquals(0L, split.seed());
        assertTrue(split.hasValidationExamples());
        assertTrue(split.hasTestExamples());
        assertExampleIndices(new int[] {0, 1}, split.trainExamples());
        assertExampleIndices(new int[] {2}, split.validationExamples());
        assertExampleIndices(new int[] {3, 4}, split.testExamples());
    }

    @Test
    void splitsByFloorFractions() {
        DiscreteTokenDatasetSplit split = DiscreteTokenDatasetSplitter.sequentialByFractions(examples(10), 0.2d, 0.3d);

        assertEquals(5, split.trainCount());
        assertEquals(2, split.validationCount());
        assertEquals(3, split.testCount());
        assertExampleIndices(new int[] {0, 1, 2, 3, 4}, split.trainExamples());
        assertExampleIndices(new int[] {5, 6}, split.validationExamples());
        assertExampleIndices(new int[] {7, 8, 9}, split.testExamples());
    }

    @Test
    void shuffledSplitsAreDeterministicBySeed() {
        DiscreteTokenDatasetSplit first = DiscreteTokenDatasetSplitter.shuffledByCounts(examples(8), 2, 2, 99L);
        DiscreteTokenDatasetSplit second = DiscreteTokenDatasetSplitter.shuffledByCounts(examples(8), 2, 2, 99L);

        assertTrue(first.shuffled());
        assertEquals(99L, first.seed());
        assertExampleIndices(exampleIndices(first.trainExamples()), second.trainExamples());
        assertExampleIndices(exampleIndices(first.validationExamples()), second.validationExamples());
        assertExampleIndices(exampleIndices(first.testExamples()), second.testExamples());
    }

    @Test
    void stratifiedSequentialFractionsPreserveTaskMix() {
        DiscreteTokenDatasetSplit split =
                DiscreteTokenDatasetSplitter.stratifiedSequentialByFractions(mixedExamples(), 0.25d, 0.25d);

        assertEquals(4, split.trainCount());
        assertEquals(2, split.validationCount());
        assertEquals(2, split.testCount());
        assertFalse(split.shuffled());
        assertExampleIndices(new int[] {0, 1, 10, 11}, split.trainExamples());
        assertExampleIndices(new int[] {2, 12}, split.validationExamples());
        assertExampleIndices(new int[] {3, 13}, split.testExamples());
        assertTaskCount(2, "graph-coloring", split.trainProfile());
        assertTaskCount(2, "nqueens", split.trainProfile());
        assertTaskCount(1, "graph-coloring", split.validationProfile());
        assertTaskCount(1, "nqueens", split.validationProfile());
        assertTaskCount(1, "graph-coloring", split.testProfile());
        assertTaskCount(1, "nqueens", split.testProfile());
    }

    @Test
    void stratifiedShuffledFractionsAreDeterministicBySeed() {
        DiscreteTokenDatasetSplit first =
                DiscreteTokenDatasetSplitter.stratifiedShuffledByFractions(mixedExamples(), 0.25d, 0.25d, 1234L);
        DiscreteTokenDatasetSplit second =
                DiscreteTokenDatasetSplitter.stratifiedShuffledByFractions(mixedExamples(), 0.25d, 0.25d, 1234L);

        assertTrue(first.shuffled());
        assertEquals(1234L, first.seed());
        assertExampleIndices(exampleIndices(first.trainExamples()), second.trainExamples());
        assertExampleIndices(exampleIndices(first.validationExamples()), second.validationExamples());
        assertExampleIndices(exampleIndices(first.testExamples()), second.testExamples());
        assertTaskCount(2, "graph-coloring", first.trainProfile());
        assertTaskCount(2, "nqueens", first.trainProfile());
        assertTaskCount(1, "graph-coloring", first.validationProfile());
        assertTaskCount(1, "nqueens", first.validationProfile());
        assertTaskCount(1, "graph-coloring", first.testProfile());
        assertTaskCount(1, "nqueens", first.testProfile());
    }

    @Test
    void splitBuildsConvenienceEpochs() {
        DiscreteTokenDatasetSplit split = DiscreteTokenDatasetSplitter.sequentialByCounts(examples(5), 1, 2);

        DiscreteTokenDatasetEpoch train = split.trainEpoch(2, -10, -20, false, 0L, false);
        DiscreteTokenDatasetEpoch validation = split.validationEpoch(2, -1);
        DiscreteTokenDatasetEpoch test = split.testEpoch(2, -1);

        assertEquals(1, train.batchCount());
        assertEquals(2, train.emittedExampleCount());
        assertArrayEquals(new int[] {0, 1}, train.batches().get(0).exampleIndices());
        assertEquals(-10, train.inputPadToken());
        assertEquals(-20, train.targetPadToken());
        assertEquals(1, validation.emittedExampleCount());
        assertArrayEquals(new int[] {2}, validation.batches().get(0).exampleIndices());
        assertEquals(2, test.emittedExampleCount());
        assertArrayEquals(new int[] {3, 4}, test.batches().get(0).exampleIndices());
    }

    @Test
    void rejectsInvalidSplitInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSplitter.sequentialByCounts(examples(2), 2, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSplitter.sequentialByCounts(examples(2), -1, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSplitter.sequentialByFractions(examples(2), 0.6d, 0.5d));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSplitter.sequentialByFractions(examples(2), Double.NaN, 0.0d));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSplitter.stratifiedSequentialByFractions(examples(2), 0.6d, 0.5d));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetSplitter.sequentialByCounts(List.of(example(0), null), 0, 0));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetSplit(null, List.of(), List.of(), false, 0L));
    }

    private static List<DiscreteTokenDatasetExample> mixedExamples() {
        return List.of(
                example("graph-coloring", 0),
                example("graph-coloring", 1),
                example("graph-coloring", 2),
                example("graph-coloring", 3),
                example("nqueens", 10),
                example("nqueens", 11),
                example("nqueens", 12),
                example("nqueens", 13));
    }

    private static List<DiscreteTokenDatasetExample> examples(int count) {
        return java.util.stream.IntStream.range(0, count)
                .mapToObj(DiscreteTokenDatasetSplitterTest::example)
                .toList();
    }

    private static DiscreteTokenDatasetExample example(int index) {
        return example("task", index);
    }

    private static DiscreteTokenDatasetExample example(String taskId, int index) {
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                new int[] {index + 1},
                new int[] {index + 2},
                1,
                Map.of("index", index));
    }

    private static void assertExampleIndices(int[] expected, List<DiscreteTokenDatasetExample> examples) {
        assertArrayEquals(expected, exampleIndices(examples));
    }

    private static int[] exampleIndices(List<DiscreteTokenDatasetExample> examples) {
        return examples.stream().mapToInt(DiscreteTokenDatasetExample::exampleIndex).toArray();
    }

    private static void assertTaskCount(int expected, String taskId, DiscreteTokenDatasetProfile profile) {
        assertEquals(expected, profile.taskExampleCounts().get(taskId));
    }
}
