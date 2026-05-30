package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetPlanReadinessGateTest {
    @Test
    void trainingGateAcceptsReadyPlansEvenWhenWarningsExist() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReadinessReport report =
                DiscreteTokenDatasetPlanReadinessGate.training().evaluate(plan);

        assertTrue(report.accepted());
        assertFalse(report.blockedByWarnings());
        assertEquals("accepted", report.gateStatus());
        assertEquals("dataset plan accepted", report.summary());
        assertTrue(report.diagnostics().hasWarnings());
        assertTrue(report.rejectionReasons().isEmpty());
    }

    @Test
    void strictGateBlocksReadyPlansWithWarnings() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReadinessReport report =
                DiscreteTokenDatasetPlanReadinessGate.strict().evaluate(plan);

        assertFalse(report.accepted());
        assertTrue(report.blockedByWarnings());
        assertEquals("warning-blocked", report.gateStatus());
        assertTrue(report.rejectionReasons().contains("validation split is empty"));
        assertTrue(report.summary().contains("dataset plan warning-blocked"));
        assertThrows(IllegalStateException.class, report::requireAccepted);
    }

    @Test
    void lenientGateSuppressesOptionalWarningsBeforeEvaluating() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReadinessReport report =
                DiscreteTokenDatasetPlanReadinessGate.lenient().evaluate(plan);

        assertTrue(report.accepted());
        assertFalse(report.diagnostics().hasWarnings());
        assertEquals("accepted", report.gateStatus());
    }

    @Test
    void readinessHelperUsesProvidedGate() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReadinessReport report =
                plan.readiness(DiscreteTokenDatasetPlanReadinessGate.strict());

        assertFalse(report.accepted());
        assertEquals("warning-blocked", report.gateStatus());
        assertThrows(NullPointerException.class, () -> plan.readiness(null));
    }

    @Test
    void readinessGateAlwaysRejectsBlockedTrainingPlans() {
        DiscreteTokenDatasetPlan plan = emptyTrainingPlan();

        DiscreteTokenDatasetPlanReadinessReport report =
                DiscreteTokenDatasetPlanReadinessGate.lenient().evaluate(plan);

        assertFalse(report.accepted());
        assertFalse(report.blockedByWarnings());
        assertEquals("blocked", report.gateStatus());
        assertTrue(report.rejectionReasons().contains("dataset is not ready for training"));
        assertTrue(report.rejectionReasons().contains("train split is empty"));
    }

    @Test
    void requireReadyReturnsDiagnosticsWhenAccepted() {
        DiscreteTokenDatasetPlan plan = cleanPlan();

        DiscreteTokenDatasetPlanDiagnostics diagnostics =
                DiscreteTokenDatasetPlanReadinessGate.strict().requireReady(plan);

        assertFalse(diagnostics.hasWarnings());
        assertTrue(diagnostics.isReadyForTraining());
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetPlanReadinessReport report =
                DiscreteTokenDatasetPlanReadinessGate.strict().evaluate(warningHeavyPlan());

        Map<String, Object> metadata = report.toMetadata();

        assertEquals("warning-blocked", metadata.get("gateStatus"));
        assertEquals(false, metadata.get("accepted"));
        assertEquals(true, metadata.get("failOnWarnings"));
        assertEquals(true, metadata.get("blockedByWarnings"));
        assertTrue(((List<?>) metadata.get("rejectionReasons")).contains("validation split is empty"));
        assertTrue(((Map<?, ?>) metadata.get("diagnostics")).containsKey("status"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void readinessGateAndDiagnosticsPolicyRoundTripMetadata() {
        DiscreteTokenDatasetPlanReadinessGate gate =
                new DiscreteTokenDatasetPlanReadinessGate(
                        DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                                .withHighPaddingRateThreshold(0.42d)
                                .withEvaluationSplitWarnings(false)
                                .withKnownSolutionWarnings(false),
                        true);

        DiscreteTokenDatasetPlanReadinessGate parsed =
                DiscreteTokenDatasetPlanReadinessGate.fromMetadata(gate.toMetadata());

        assertEquals(gate, parsed);
        assertEquals(gate.toMetadata(), parsed.toMetadata());
        assertEquals(gate.diagnosticsPolicy(),
                DiscreteTokenDatasetPlanDiagnosticsPolicy.fromMetadata(gate.diagnosticsPolicy().toMetadata()));
        assertThrows(UnsupportedOperationException.class, () -> gate.toMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> gate.diagnosticsPolicy().toMetadata().put("bad", "value"));
    }

    @Test
    void readinessGateMetadataAcceptsJsonLikeScalarValues() {
        Map<String, Object> diagnosticsPolicyMetadata = new java.util.LinkedHashMap<>(
                DiscreteTokenDatasetPlanDiagnosticsPolicy.lenient().toMetadata());
        diagnosticsPolicyMetadata.put("highPaddingRateThreshold", "0.25");
        diagnosticsPolicyMetadata.put("warnOnHighPaddingRate", "true");
        Map<String, Object> gateMetadata = Map.of(
                "diagnosticsPolicy", diagnosticsPolicyMetadata,
                "failOnWarnings", "true");

        DiscreteTokenDatasetPlanReadinessGate gate =
                DiscreteTokenDatasetPlanReadinessGate.fromMetadata(gateMetadata);

        assertTrue(gate.failOnWarnings());
        assertEquals(0.25d, gate.diagnosticsPolicy().highPaddingRateThreshold());
        assertTrue(gate.diagnosticsPolicy().warnOnHighPaddingRate());
    }

    @Test
    void rejectsMalformedGateInputs() {
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetPlanReadinessGate(null, false));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetPlanReadinessReport(null, false));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanReadinessGate.training()
                        .evaluate((DiscreteTokenDatasetPlan) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanReadinessGate.training()
                        .evaluate((DiscreteTokenDatasetPlanDiagnostics) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanReadinessGate.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanReadinessGate.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanDiagnosticsPolicy.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanDiagnosticsPolicy.fromMetadata(
                        Map.of(
                                "highPaddingRateThreshold", 2.0d,
                                "warnOnMissingValidationSplit", true,
                                "warnOnMissingTestSplit", true,
                                "warnOnDroppedTrainingExamples", true,
                                "warnOnMissingKnownSolutionCounts", true,
                                "warnOnPartialKnownSolutionCoverage", true,
                                "warnOnHighPaddingRate", true,
                                "warnOnMissingTaskTrainCoverage", true,
                                "warnOnMissingTaskValidationCoverage", true,
                                "warnOnMissingTaskTestCoverage", true)));
    }

    private static DiscreteTokenDatasetPlan cleanPlan() {
        return DiscreteTokenDatasetPlanner.plan(
                List.of(
                        knownExample("graph-coloring", 0, 1),
                        knownExample("graph-coloring", 1, 2),
                        knownExample("graph-coloring", 2, 1),
                        knownExample("graph-coloring", 3, 2),
                        knownExample("nqueens", 10, 1),
                        knownExample("nqueens", 11, 2),
                        knownExample("nqueens", 12, 1),
                        knownExample("nqueens", 13, 2)),
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

    private static DiscreteTokenDatasetPlan emptyTrainingPlan() {
        return DiscreteTokenDatasetPlanner.plan(
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
}
