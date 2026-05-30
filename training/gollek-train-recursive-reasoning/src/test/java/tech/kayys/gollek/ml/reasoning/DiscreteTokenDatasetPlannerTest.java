package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetPlannerTest {
    @Test
    void plansStratifiedLengthSortedDatasetFlow() {
        DiscreteTokenDatasetPlanConfig config = new DiscreteTokenDatasetPlanConfig(
                0.25d,
                0.25d,
                DiscreteTokenDatasetSplitMode.STRATIFIED_SEQUENTIAL_FRACTIONS,
                7L,
                2,
                4,
                -10,
                -20,
                DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                99L,
                false);

        DiscreteTokenDatasetPlan plan = DiscreteTokenDatasetPlanner.plan(mixedExamples(), config);

        assertEquals(config, plan.config());
        assertEquals(8, plan.profile().exampleCount());
        assertEquals(4, plan.split().trainCount());
        assertEquals(2, plan.split().validationCount());
        assertEquals(2, plan.split().testCount());
        assertTaskCount(2, "graph-coloring", plan.split().trainProfile());
        assertTaskCount(2, "nqueens", plan.split().trainProfile());
        assertEquals(2, plan.trainEpoch().batchCount());
        assertEquals(1, plan.validationEpoch().batchCount());
        assertEquals(1, plan.testEpoch().batchCount());
        assertTrue(plan.hasValidationEpoch());
        assertTrue(plan.hasTestEpoch());
        assertEquals(4L, plan.emittedTrainingExamples());
        assertEquals(plan.trainEpoch().paddingRate(), plan.trainPaddingRate(), 1e-9);
        assertEquals(-10, plan.trainEpoch().inputPadToken());
        assertEquals(-20, plan.trainEpoch().targetPadToken());
        assertEquals(DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED, plan.config().trainEpochMode());
        assertTrue(plan.config().usesStratifiedSplit());
        assertTrue(plan.config().usesLengthSortedTraining());
        assertFalse(plan.config().usesShuffledSplit());

        assertArrayEquals(new int[] {0, 10}, plan.trainEpoch().batches().get(0).exampleIndices());
        assertArrayEquals(new int[] {1, 11}, plan.trainEpoch().batches().get(1).exampleIndices());
        assertArrayEquals(new int[] {2, 12}, plan.validationEpoch().batches().get(0).exampleIndices());
        assertArrayEquals(new int[] {3, 13}, plan.testEpoch().batches().get(0).exampleIndices());
    }

    @Test
    void shuffledPlanIsDeterministicBySeeds() {
        DiscreteTokenDatasetPlanConfig config = new DiscreteTokenDatasetPlanConfig(
                0.25d,
                0.25d,
                DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                1234L,
                2,
                2,
                -1,
                -1,
                DiscreteTokenDatasetTrainEpochMode.SHUFFLED,
                4321L,
                false);

        DiscreteTokenDatasetPlan first = DiscreteTokenDatasetPlanner.plan(mixedExamples(), config);
        DiscreteTokenDatasetPlan second = DiscreteTokenDatasetPlanner.plan(mixedExamples(), config);

        assertTrue(first.config().usesShuffledSplit());
        assertTrue(first.split().shuffled());
        assertTrue(first.trainEpoch().shuffled());
        assertEquals(4321L, first.trainEpoch().seed());
        assertArrayEquals(
                first.trainEpoch().batches().get(0).exampleIndices(),
                second.trainEpoch().batches().get(0).exampleIndices());
        assertArrayEquals(
                first.validationEpoch().batches().get(0).exampleIndices(),
                second.validationEpoch().batches().get(0).exampleIndices());
        assertArrayEquals(
                first.testEpoch().batches().get(0).exampleIndices(),
                second.testEpoch().batches().get(0).exampleIndices());
    }

    @Test
    void convenienceConfigUsesStratifiedLengthSortedDefaults() {
        DiscreteTokenDatasetPlanConfig config =
                DiscreteTokenDatasetPlanConfig.stratifiedLengthSorted(0.1d, 0.1d, 17L, 8, 16, -1);

        assertEquals(DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS, config.splitMode());
        assertEquals(DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED, config.trainEpochMode());
        assertEquals(17L, config.splitSeed());
        assertEquals(0L, config.trainEpochSeed());
        assertEquals(8, config.trainBatchSize());
        assertEquals(16, config.evaluationBatchSize());
        assertEquals(-1, config.inputPadToken());
        assertEquals(-1, config.targetPadToken());
        assertTrue(config.usesStratifiedSplit());
        assertTrue(config.usesShuffledSplit());
        assertTrue(config.usesLengthSortedTraining());
        assertFalse(config.dropLastTrain());
    }

    @Test
    void rejectsInvalidPlanInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanConfig(
                        0.6d,
                        0.5d,
                        DiscreteTokenDatasetSplitMode.SHUFFLED_FRACTIONS,
                        0L,
                        1,
                        1,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                        0L,
                        false));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanConfig(
                        0.1d,
                        0.1d,
                        DiscreteTokenDatasetSplitMode.SHUFFLED_FRACTIONS,
                        0L,
                        0,
                        1,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                        0L,
                        false));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanner.plan(null, validConfig()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanner.plan(mixedExamples(), null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlan(
                        validConfig(),
                        DiscreteTokenDatasetProfiler.profile(mixedExamples()),
                        DiscreteTokenDatasetSplitter.sequentialByFractions(mixedExamples(), 0.25d, 0.25d),
                        DiscreteTokenDatasetEpochBatcher.sequential(List.of(example("bad", 99, 1)), 1, -1),
                        DiscreteTokenDatasetEpochBatcher.sequential(List.of(), 1, -1),
                        DiscreteTokenDatasetEpochBatcher.sequential(List.of(), 1, -1)));
    }

    private static DiscreteTokenDatasetPlanConfig validConfig() {
        return new DiscreteTokenDatasetPlanConfig(
                0.25d,
                0.25d,
                DiscreteTokenDatasetSplitMode.SEQUENTIAL_FRACTIONS,
                0L,
                2,
                2,
                -1,
                -1,
                DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                0L,
                false);
    }

    private static List<DiscreteTokenDatasetExample> mixedExamples() {
        return List.of(
                example("graph-coloring", 0, 1),
                example("graph-coloring", 1, 3),
                example("graph-coloring", 2, 2),
                example("graph-coloring", 3, 4),
                example("nqueens", 10, 1),
                example("nqueens", 11, 3),
                example("nqueens", 12, 2),
                example("nqueens", 13, 4));
    }

    private static DiscreteTokenDatasetExample example(String taskId, int index, int inputLength) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                1,
                Map.of("inputLength", inputLength));
    }

    private static void assertTaskCount(int expected, String taskId, DiscreteTokenDatasetProfile profile) {
        assertEquals(expected, profile.taskExampleCounts().get(taskId));
    }
}
