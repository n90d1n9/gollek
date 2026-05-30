package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumeReportTest {
    @Test
    void acceptsMatchingCurrentCheckpointAndPlan() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReport planReport = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetPlanReport defaultResumeReport = plan.report();
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(planReport).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);

        DiscreteTokenDatasetCheckpointResumeReport resumeReport = snapshot.resumeReport(plan);

        assertTrue(resumeReport.ready());
        assertEquals("ready", resumeReport.status());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanChecked());
        assertFalse(resumeReport.policyTracked());
        assertTrue(resumeReport.currentPlanAccepted());
        assertEquals("accepted", resumeReport.currentPlanGateStatus());
        assertTrue(resumeReport.rejectionReasons().isEmpty());
        assertEquals(
                "checkpoint resume ready: run-001 step 12 dataset " + planReport.fingerprint().shortValue(),
                resumeReport.summary());
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("ready", explanation.status());
        assertEquals("resume-ready", explanation.primaryCode());
        assertEquals("ready", explanation.primaryCategory());
        assertEquals(1, explanation.findings().size());
        assertEquals("resume-ready", explanation.toMetadata().get("primaryCode"));
        assertEquals(true, resumeReport.toMetadata().get("ready"));
        assertEquals(true, resumeReport.toMetadata().get("expectationAccepted"));
        assertEquals(true, resumeReport.toMetadata().get("currentPlanChecked"));
        assertEquals(true, resumeReport.toMetadata().get("currentPlanAccepted"));
        assertEquals("accepted", resumeReport.toMetadata().get("currentPlanGateStatus"));
        assertEquals(false, resumeReport.toMetadata().get("policyTracked"));
        assertEquals("matched", ((Map<?, ?>) resumeReport.toMetadata().get("fingerprintMatch")).get("status"));
        assertEquals(false, ((Map<?, ?>) resumeReport.toMetadata().get("expectation")).get("active"));
        assertEquals(
                defaultResumeReport.toMetadata(),
                ((Map<?, ?>) resumeReport.toMetadata().get("currentPlanReport")));
        assertEquals(manifest.toMetadata(), ((Map<?, ?>) resumeReport.toMetadata().get("checkpoint")));
        resumeReport.requireReady();
    }

    @Test
    void acceptsMatchingCurrentCheckpointAndReport() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(report).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), report);

        assertTrue(resumeReport.ready());
        assertTrue(resumeReport.fingerprintMatched());
    }

    @Test
    void blocksWhenCurrentReportFailsEvenIfCheckpointWasAccepted() {
        DiscreteTokenDatasetPlan warningPlan = warningHeavyPlan();
        DiscreteTokenDatasetPlanReport checkpointReport =
                warningPlan.report(DiscreteTokenDatasetPlanReadinessGate.training());
        DiscreteTokenDatasetPlanReport currentStrictReport =
                warningPlan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(checkpointReport).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        currentStrictReport);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanChecked());
        assertFalse(resumeReport.currentPlanAccepted());
        assertEquals("warning-blocked", resumeReport.currentPlanGateStatus());
        assertTrue(resumeReport.rejectionReasons()
                .contains("current dataset plan was not accepted: warning-blocked"));
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("blocked", explanation.status());
        assertEquals("current-plan-rejected", explanation.primaryCode());
        assertEquals("current-dataset", explanation.primaryCategory());
        assertTrue(explanation.nextSteps().get(0).contains("Fix the current dataset readiness issues"));
        assertEquals("warning-blocked", resumeReport.toMetadata().get("currentPlanGateStatus"));
        assertEquals(currentStrictReport.toMetadata(), resumeReport.toMetadata().get("currentPlanReport"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void acceptsMatchingResumeExpectation() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("gram-nqueens")
                        .runId("run-001")
                        .modelFamily("gram")
                        .seed(2026L)
                        .minimumCheckpointStep(12L)
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        plan,
                        expectation);

        assertTrue(resumeReport.ready());
        assertTrue(resumeReport.expectationAccepted());
        assertEquals(expectation.toMetadata(), resumeReport.expectation().toMetadata());
        resumeReport.requireReady();
    }

    @Test
    void blocksWhenResumeExpectationDiffers() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("other-experiment")
                        .runId("run-999")
                        .modelFamily("other-family")
                        .seed(7L)
                        .minimumCheckpointStep(20L)
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest.toMetadata(),
                        plan,
                        expectation);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertFalse(resumeReport.expectationAccepted());
        assertTrue(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("experimentName expected other-experiment")));
        assertTrue(resumeReport.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("checkpointStep expected >= 20")));
        assertEquals(false, resumeReport.toMetadata().get("expectationAccepted"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void blocksWhenCheckpointSchemaIsNotCurrent() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()))
                        .schemaVersion("gollek.discrete-token-checkpoint-manifest.v2")
                        .build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), plan);

        assertFalse(resumeReport.ready());
        assertFalse(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().get(0).contains("schema"));
        assertThrows(IllegalStateException.class, resumeReport::requireReady);
    }

    @Test
    void blocksWhenCheckpointDatasetWasRejected() {
        DiscreteTokenDatasetPlanReport report =
                warningHeavyPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(report).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), report);

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertFalse(resumeReport.datasetAccepted());
        assertTrue(resumeReport.fingerprintMatched());
        assertFalse(resumeReport.currentPlanAccepted());
        assertTrue(resumeReport.rejectionReasons().contains("checkpoint dataset was not accepted: warning-blocked"));
        assertTrue(resumeReport.rejectionReasons()
                .contains("current dataset plan was not accepted: warning-blocked"));
        assertTrue(resumeReport.summary().contains("warning-blocked"));
    }

    @Test
    void blocksWhenCurrentPlanFingerprintDiffers() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(manifest.toMetadata(), changedPlan());

        assertFalse(resumeReport.ready());
        assertTrue(resumeReport.schemaAccepted());
        assertTrue(resumeReport.datasetAccepted());
        assertFalse(resumeReport.fingerprintMatched());
        assertTrue(resumeReport.currentPlanAccepted());
        assertEquals("mismatched", resumeReport.fingerprintMatch().status());
        assertTrue(resumeReport.rejectionReasons().get(0).contains("dataset fingerprint mismatched"));
        DiscreteTokenDatasetCheckpointResumeExplanation explanation = resumeReport.explanation();
        assertEquals("dataset-fingerprint-mismatch", explanation.primaryCode());
        assertEquals("dataset-fingerprint", explanation.primaryCategory());
        assertEquals("error", explanation.findings().get(0).severity());
        assertTrue(((List<?>) explanation.findings().get(0).details().get("mismatchReasons"))
                .contains("fingerprint value differs"));
        assertTrue(resumeReport.summary().contains("blocked"));
    }

    @Test
    void rejectsMalformedInputs() {
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(
                        manifest(cleanPlan().report()).build());

        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(null, snapshot.verifyPlan(cleanPlan())));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(snapshot, null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(null, cleanPlan()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(null, cleanPlan()));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(snapshot, (DiscreteTokenDatasetPlan) null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(snapshot, (DiscreteTokenDatasetPlanReport) null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                        cleanPlan().report(),
                        mapWithNullPolicyValue()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetCheckpointResumeReport(
                        snapshot,
                        snapshot.verifyPlan(cleanPlan()),
                        DiscreteTokenDatasetCheckpointResumeExpectation.none(),
                        cleanPlan().report(),
                        mapWithBlankPolicyKey()));
    }

    @Test
    void exportsImmutableMetadata() {
        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumeReport.fromMetadata(
                        manifest(cleanPlan().report()).build().toMetadata(),
                        cleanPlan());

        assertThrows(UnsupportedOperationException.class, () -> resumeReport.toMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> resumeReport.withPolicyMetadata(Map.of("source", "unit-test"))
                        .policyMetadata()
                        .put("bad", "value"));
    }

    private static Map<String, Object> mapWithNullPolicyValue() {
        java.util.HashMap<String, Object> values = new java.util.HashMap<>();
        values.put("bad", null);
        return values;
    }

    private static Map<String, Object> mapWithBlankPolicyKey() {
        java.util.HashMap<String, Object> values = new java.util.HashMap<>();
        values.put(" ", "bad");
        return values;
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-nqueens")
                .runId("run-001")
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(1_900_000_000_000L)
                .createdBy("trainer-test");
    }

    private static DiscreteTokenDatasetPlan cleanPlan() {
        return plan(0L);
    }

    private static DiscreteTokenDatasetPlan changedPlan() {
        return plan(42L);
    }

    private static DiscreteTokenDatasetPlan plan(long seed) {
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
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        seed,
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
