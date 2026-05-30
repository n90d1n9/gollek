package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetPlanDiagnosticsTest {
    @Test
    void summarizesPlannedDatasetHealth() {
        DiscreteTokenDatasetPlan plan = DiscreteTokenDatasetPlanner.plan(
                mixedExamples(),
                new DiscreteTokenDatasetPlanConfig(
                        0.25d,
                        0.25d,
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SEQUENTIAL_FRACTIONS,
                        0L,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                        0L,
                        false));

        DiscreteTokenDatasetPlanDiagnostics diagnostics = plan.diagnostics();

        assertEquals(8, diagnostics.exampleCount());
        assertEquals(2, diagnostics.taskCount());
        assertEquals(4, diagnostics.trainCount());
        assertEquals(2, diagnostics.validationCount());
        assertEquals(2, diagnostics.testCount());
        assertEquals(2, diagnostics.trainBatchCount());
        assertEquals(1, diagnostics.validationBatchCount());
        assertEquals(1, diagnostics.testBatchCount());
        assertEquals(4L, diagnostics.emittedTrainingExamples());
        assertEquals(0L, diagnostics.droppedTrainingExamples());
        assertEquals(1.0d, diagnostics.knownSolutionCoverageRate(), 1e-9);
        assertEquals(plan.trainEpoch().paddingRate(), diagnostics.trainPaddingRate(), 1e-9);
        assertEquals(plan.trainEpoch().paddingTokenCount(), diagnostics.trainPaddingTokenCount());
        assertEquals(plan.trainEpoch().paddedTokenCapacity(), diagnostics.trainPaddedTokenCapacity());
        assertTrue(diagnostics.isReadyForTraining());
        assertTrue(diagnostics.hasValidationSet());
        assertTrue(diagnostics.hasTestSet());
        assertTrue(diagnostics.hasKnownSolutionCoverage());
        assertFalse(diagnostics.hasWarnings());
        assertEquals("ready", diagnostics.status());
        assertEquals(2, diagnostics.taskSplits().size());

        DiscreteTokenDatasetTaskSplitDiagnostics graphSplit = diagnostics.taskSplits().get(0);
        assertEquals("graph-coloring", graphSplit.taskId());
        assertEquals(4, graphSplit.exampleCount());
        assertEquals(2, graphSplit.trainCount());
        assertEquals(1, graphSplit.validationCount());
        assertEquals(1, graphSplit.testCount());
        assertEquals(0.5d, graphSplit.trainRate(), 1e-9);
        assertEquals(0.25d, graphSplit.validationRate(), 1e-9);
        assertEquals(0.25d, graphSplit.testRate(), 1e-9);
        assertTrue(graphSplit.hasTrainExamples());
        assertTrue(graphSplit.hasValidationExamples());
        assertTrue(graphSplit.hasTestExamples());

        Map<String, Object> metadata = diagnostics.toMetadata();
        assertEquals("ready", metadata.get("status"));
        assertEquals(true, metadata.get("readyForTraining"));
        assertEquals(false, metadata.get("hasWarnings"));
        assertEquals(8, metadata.get("exampleCount"));
        assertEquals(2, metadata.get("taskCount"));
        assertEquals(4, ((Map<?, ?>) metadata.get("split")).get("trainCount"));
        assertEquals(2, ((Map<?, ?>) metadata.get("split")).get("validationCount"));
        assertEquals(1, ((Map<?, ?>) metadata.get("batches")).get("validationBatchCount"));
        assertEquals(4L, ((Map<?, ?>) metadata.get("trainingExamples")).get("emittedTrainingExamples"));
        assertEquals(0L, ((Map<?, ?>) metadata.get("trainingExamples")).get("droppedTrainingExamples"));
        assertEquals(1.0d, (Double) metadata.get("knownSolutionCoverageRate"), 1e-9);
        assertEquals(true, metadata.get("knownSolutionCoverageAvailable"));
        assertEquals(diagnostics.trainPaddingRate(), (Double) ((Map<?, ?>) metadata.get("trainPadding")).get("rate"), 1e-9);
        List<?> taskSplits = (List<?>) metadata.get("taskSplits");
        assertEquals(2, taskSplits.size());
        assertEquals("graph-coloring", ((Map<?, ?>) taskSplits.get(0)).get("taskId"));
        assertEquals(2, ((Map<?, ?>) taskSplits.get(0)).get("trainCount"));
        assertEquals(diagnostics.warnings(), metadata.get("warnings"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> metadata.put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> putRaw(metadata.get("split"), "bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> putRaw(taskSplits.get(0), "bad", "value"));

        DiscreteTokenDatasetPlanDiagnostics reloaded =
                DiscreteTokenDatasetPlanDiagnostics.fromMetadata(metadata);
        assertEquals(diagnostics, reloaded);
        assertEquals(graphSplit, DiscreteTokenDatasetTaskSplitDiagnostics.fromMetadata(
                graphSplit.toMetadata()));
    }

    @Test
    void warnsAboutEmptyEvaluationSplitsUnknownSolutionsDroppedExamplesAndPadding() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanDiagnostics diagnostics =
                DiscreteTokenDatasetPlanDiagnostics.from(plan, 0.10d);

        assertTrue(diagnostics.hasWarnings());
        assertEquals("warning", diagnostics.status());
        assertTrue(diagnostics.isReadyForTraining());
        assertFalse(diagnostics.hasValidationSet());
        assertFalse(diagnostics.hasTestSet());
        assertFalse(diagnostics.hasKnownSolutionCoverage());
        assertEquals(2L, diagnostics.emittedTrainingExamples());
        assertEquals(1L, diagnostics.droppedTrainingExamples());
        assertTrue(diagnostics.warnings().contains("validation split is empty"));
        assertTrue(diagnostics.warnings().contains("test split is empty"));
        assertTrue(diagnostics.warnings().contains("dataset has no known solution counts"));
        assertTrue(diagnostics.warnings().contains("train epoch dropped 1 example(s)"));
        assertTrue(diagnostics.warnings().stream().anyMatch(warning -> warning.startsWith("train padding rate is high:")));
        assertThrows(
                UnsupportedOperationException.class,
                () -> diagnostics.warnings().add("bad"));
    }

    @Test
    void lenientPolicySuppressesOptionalWarnings() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanDiagnostics diagnostics =
                plan.diagnostics(DiscreteTokenDatasetPlanDiagnosticsPolicy.lenient());

        assertFalse(diagnostics.hasWarnings());
        assertEquals("ready", diagnostics.status());
        assertTrue(diagnostics.isReadyForTraining());
        assertFalse(diagnostics.hasValidationSet());
        assertFalse(diagnostics.hasTestSet());
        assertEquals(1L, diagnostics.droppedTrainingExamples());
        assertFalse(diagnostics.hasKnownSolutionCoverage());
    }

    @Test
    void customPolicyCanKeepOnlyPaddingWarning() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();
        DiscreteTokenDatasetPlanDiagnosticsPolicy policy =
                DiscreteTokenDatasetPlanDiagnosticsPolicy.lenient()
                        .withPaddingWarning(true)
                        .withHighPaddingRateThreshold(0.10d);

        DiscreteTokenDatasetPlanDiagnostics diagnostics =
                DiscreteTokenDatasetPlanDiagnostics.from(plan, policy);

        assertEquals(1, diagnostics.warnings().size());
        assertTrue(diagnostics.warnings().get(0).startsWith("train padding rate is high:"));
        assertEquals("warning", diagnostics.status());
    }

    @Test
    void warnsWhenTaskIsMissingFromASplit() {
        DiscreteTokenDatasetPlan plan = DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("alpha", 0, 1),
                        knownExample("alpha", 1, 1),
                        knownExample("beta", 2, 1),
                        knownExample("beta", 3, 1)),
                new DiscreteTokenDatasetPlanConfig(
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
                        false));

        DiscreteTokenDatasetPlanDiagnostics diagnostics = plan.diagnostics();

        assertTrue(diagnostics.warnings().contains("train split is missing task: beta"));
        assertTrue(diagnostics.warnings().contains("validation split is missing task: alpha"));
        assertTrue(diagnostics.warnings().contains("test split is missing task: alpha"));
        assertEquals("warning", diagnostics.status());
        assertEquals(0, diagnostics.taskSplits().get(0).validationCount());
        assertEquals(0, diagnostics.taskSplits().get(0).testCount());
        assertEquals(0, diagnostics.taskSplits().get(1).trainCount());
    }

    @Test
    void policyCanSuppressPerTaskCoverageWarnings() {
        DiscreteTokenDatasetPlan plan = DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("alpha", 0, 1),
                        knownExample("alpha", 1, 1),
                        knownExample("beta", 2, 1),
                        knownExample("beta", 3, 1)),
                new DiscreteTokenDatasetPlanConfig(
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
                        false));
        DiscreteTokenDatasetPlanDiagnosticsPolicy policy =
                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                        .withPerTaskSplitWarnings(false);

        DiscreteTokenDatasetPlanDiagnostics diagnostics = plan.diagnostics(policy);

        assertFalse(diagnostics.hasWarnings());
        assertEquals("ready", diagnostics.status());
        assertEquals(2, diagnostics.taskSplits().size());
    }

    @Test
    void warnsWhenTrainingSplitIsEmpty() {
        DiscreteTokenDatasetPlan plan = DiscreteTokenDatasetPlanner.plan(
                List.of(knownExample(0, 1), knownExample(1, 1)),
                new DiscreteTokenDatasetPlanConfig(
                        1.0d,
                        0.0d,
                        DiscreteTokenDatasetSplitMode.SEQUENTIAL_FRACTIONS,
                        0L,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                        0L,
                        false));

        DiscreteTokenDatasetPlanDiagnostics diagnostics = plan.diagnostics();

        assertFalse(diagnostics.isReadyForTraining());
        assertEquals("blocked", diagnostics.status());
        assertTrue(diagnostics.warnings().contains("train split is empty"));
    }

    @Test
    void rejectsMalformedDiagnostics() {
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanDiagnostics.from(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanDiagnostics.from(
                        DiscreteTokenDatasetPlanner.plan(mixedExamples(), validConfig()),
                        Double.NaN));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanDiagnostics.from(
                        DiscreteTokenDatasetPlanner.plan(mixedExamples(), validConfig()),
                        (DiscreteTokenDatasetPlanDiagnosticsPolicy) null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                        .withHighPaddingRateThreshold(Double.POSITIVE_INFINITY));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanDiagnostics(
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1L,
                        0L,
                        1.0d,
                        0.0d,
                        0L,
                        1L,
                        List.of(new DiscreteTokenDatasetTaskSplitDiagnostics("task", 2, 1, 1, 0)),
                        List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanDiagnostics(
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1L,
                        0L,
                        1.0d,
                        0.0d,
                        0L,
                        1L,
                        List.of(new DiscreteTokenDatasetTaskSplitDiagnostics("task", 1, 1, 0, 0)),
                        List.of(" ")));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanDiagnostics(
                        1,
                        1,
                        1,
                        0,
                        0,
                        1,
                        0,
                        0,
                        1L,
                        0L,
                        1.0d,
                        0.0d,
                        0L,
                        1L,
                        List.of(),
                        List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTaskSplitDiagnostics("task", 2, 1, 0, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTaskSplitDiagnostics(" ", 0, 0, 0, 0));
    }

    private static DiscreteTokenDatasetPlan warningHeavyPlan() {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        unknownExample(0, 1),
                        unknownExample(1, 8),
                        unknownExample(2, 2)),
                new DiscreteTokenDatasetPlanConfig(
                        0.0d,
                        0.0d,
                        DiscreteTokenDatasetSplitMode.SEQUENTIAL_FRACTIONS,
                        0L,
                        2,
                        2,
                        -1,
                        -1,
                        DiscreteTokenDatasetTrainEpochMode.SEQUENTIAL,
                        0L,
                        true));
    }

    private static DiscreteTokenDatasetPlanConfig validConfig() {
        return new DiscreteTokenDatasetPlanConfig(
                0.25d,
                0.25d,
                DiscreteTokenDatasetSplitMode.STRATIFIED_SEQUENTIAL_FRACTIONS,
                0L,
                2,
                2,
                -1,
                -1,
                DiscreteTokenDatasetTrainEpochMode.LENGTH_SORTED,
                0L,
                false);
    }

    private static List<DiscreteTokenDatasetExample> mixedExamples() {
        return List.of(
                knownExample("graph-coloring", 0, 1),
                knownExample("graph-coloring", 1, 2),
                knownExample("graph-coloring", 2, 1),
                knownExample("graph-coloring", 3, 2),
                knownExample("nqueens", 10, 1),
                knownExample("nqueens", 11, 2),
                knownExample("nqueens", 12, 1),
                knownExample("nqueens", 13, 2));
    }

    private static DiscreteTokenDatasetExample knownExample(int index, int inputLength) {
        return knownExample("task", index, inputLength);
    }

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
        return example(taskId, index, inputLength, 1);
    }

    private static DiscreteTokenDatasetExample unknownExample(int index, int inputLength) {
        return example("task", index, inputLength, -1);
    }

    private static DiscreteTokenDatasetExample example(
            String taskId,
            int index,
            int inputLength,
            int knownSolutionCount) {
        int[] input = new int[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = index + i + 1;
        }
        return new DiscreteTokenDatasetExample(
                taskId,
                index,
                input,
                new int[] {index + 100},
                knownSolutionCount,
                Map.of("inputLength", inputLength));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static void putRaw(Object map, String key, Object value) {
        ((Map) map).put(key, value);
    }
}
