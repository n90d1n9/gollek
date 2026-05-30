package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetPlanReportTest {
    @Test
    void defaultReportBundlesFingerprintDiagnosticsAndTrainingReadiness() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReport report = plan.report();

        assertEquals(plan.fingerprint(), report.fingerprint());
        assertTrue(report.accepted());
        assertEquals("accepted", report.gateStatus());
        assertTrue(report.diagnostics().hasWarnings());
        assertEquals(report.diagnostics().warnings(), report.warnings());
        assertEquals("dataset plan " + report.fingerprint().shortValue() + " accepted", report.summary());
    }

    @Test
    void strictReportBlocksWarningsForCiStylePreflight() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();

        DiscreteTokenDatasetPlanReport report = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());

        assertFalse(report.accepted());
        assertEquals("warning-blocked", report.gateStatus());
        assertTrue(report.summary().contains(report.fingerprint().shortValue()));
        assertTrue(report.summary().contains("validation split is empty"));
        assertThrows(IllegalStateException.class, report::requireAccepted);
    }

    @Test
    void reportCanRequireAcceptedPlans() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());

        report.requireAccepted();

        assertTrue(report.accepted());
        assertFalse(report.diagnostics().hasWarnings());
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetPlanReport report = warningHeavyPlan()
                .report(DiscreteTokenDatasetPlanReadinessGate.strict());

        Map<String, Object> metadata = report.toMetadata();

        assertEquals(false, metadata.get("accepted"));
        assertEquals("warning-blocked", metadata.get("gateStatus"));
        assertEquals(report.fingerprint().value(), ((Map<?, ?>) metadata.get("fingerprint")).get("value"));
        assertEquals(report.readiness().gateStatus(), ((Map<?, ?>) metadata.get("readiness")).get("gateStatus"));
        assertEquals(report, DiscreteTokenDatasetPlanReport.fromMetadata(metadata));
        assertEquals(
                report.readiness(),
                DiscreteTokenDatasetPlanReadinessReport.fromMetadata((Map<?, ?>) metadata.get("readiness")));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void staticFactoryMatchesPlanHelper() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReadinessGate gate = DiscreteTokenDatasetPlanReadinessGate.strict();

        assertEquals(DiscreteTokenDatasetPlanReport.from(plan, gate), plan.report(gate));
        assertEquals(DiscreteTokenDatasetPlanReport.from(plan), plan.report());
    }

    @Test
    void rejectsMalformedReports() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report();
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetPlanReport.from(null));
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetPlanReport.from(cleanPlan(), null));
        assertThrows(NullPointerException.class, () -> cleanPlan().report(null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetPlanReport(null, report.readiness()));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetPlanReport(report.fingerprint(), null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetPlanReport.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanReport.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetPlanReadinessReport.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetPlanReport(
                        new DiscreteTokenDatasetFingerprint("SHA-256", report.fingerprint().value(), 999),
                        report.readiness()));
    }

    @Test
    void exposesOriginalReadinessReference() {
        DiscreteTokenDatasetPlanReadinessReport readiness =
                DiscreteTokenDatasetPlanReadinessGate.strict().evaluate(cleanPlan());
        DiscreteTokenDatasetPlanReport report = new DiscreteTokenDatasetPlanReport(
                cleanPlan().fingerprint(),
                readiness);

        assertSame(readiness, report.readiness());
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
