package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumePolicyTest {
    @Test
    void trainingPolicyAcceptsMatchingAcceptedCheckpointAndPlan() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.training().evaluate(manifest.toMetadata(), plan);

        assertTrue(report.ready());
        assertTrue(report.currentPlanChecked());
        assertTrue(report.policyTracked());
        assertEquals("accepted", report.currentPlanGateStatus());
        assertEquals(DiscreteTokenDatasetCheckpointResumePolicy.training().toMetadata(), report.policyMetadata());
        assertEquals(DiscreteTokenDatasetCheckpointResumePolicy.training().toMetadata(), report.toMetadata().get("policy"));
        assertEquals("strict", report.toMetadata().get("compatibilityMode"));
        assertEquals(false, report.toMetadata().get("forceAccepted"));
        assertEquals(true, report.toMetadata().get("policyTracked"));
        assertEquals(plan.report(DiscreteTokenDatasetPlanReadinessGate.training()).toMetadata(),
                report.currentPlanReport().toMetadata());
        DiscreteTokenDatasetCheckpointResumePolicy.training().requireReady(manifest, plan);
    }

    @Test
    void strictPolicyBlocksWhenCurrentPlanHasWarnings() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.training())).build();

        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest.toMetadata(), plan);

        assertFalse(report.ready());
        assertTrue(report.datasetAccepted());
        assertTrue(report.fingerprintMatched());
        assertTrue(report.policyTracked());
        assertFalse(report.currentPlanAccepted());
        assertEquals("warning-blocked", report.currentPlanGateStatus());
        assertTrue(report.rejectionReasons()
                .contains("current dataset plan was not accepted: warning-blocked"));
        assertThrows(
                IllegalStateException.class,
                () -> DiscreteTokenDatasetCheckpointResumePolicy.strict().requireReady(manifest, plan));
    }

    @Test
    void policyExpectationBlocksMismatchedCheckpointIdentity() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                                .runId("other-run")
                                .minimumCheckpointStep(99L)
                                .build());

        DiscreteTokenDatasetCheckpointResumeReport report = policy.evaluate(manifest, plan);

        assertFalse(report.ready());
        assertFalse(report.expectationAccepted());
        assertTrue(report.policyTracked());
        assertEquals(policy.toMetadata(), report.policyMetadata());
        assertTrue(report.rejectionReasons().stream().anyMatch(reason -> reason.contains("runId expected other-run")));
        assertTrue(report.rejectionReasons().stream().anyMatch(reason -> reason.contains("checkpointStep expected >= 99")));
    }

    @Test
    void snapshotOverloadUsesPolicyGateAndExpectation() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));

        DiscreteTokenDatasetCheckpointResumeReport report = snapshot.resumeReport(plan, policy);

        assertTrue(report.ready());
        assertTrue(report.expectationAccepted());
        assertTrue(report.policyTracked());
        assertEquals("accepted", report.currentPlanGateStatus());
        assertEquals(policy.expectation().toMetadata(), report.expectation().toMetadata());
        assertEquals(policy.toMetadata(), report.policyMetadata());
    }

    @Test
    void exposesImmutablePolicyMetadata() {
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                                .experimentName("gram-nqueens")
                                .build());

        Map<String, Object> metadata = policy.toMetadata();

        assertEquals(true, ((Map<?, ?>) metadata.get("currentPlanGate")).get("failOnWarnings"));
        assertEquals("gram-nqueens", ((Map<?, ?>) metadata.get("expectation")).get("experimentName"));
        assertTrue(((Map<?, ?>) ((Map<?, ?>) metadata.get("currentPlanGate")).get("diagnosticsPolicy"))
                .containsKey("warnOnHighPaddingRate"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void roundTripsPolicyMetadata() {
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withCurrentPlanGate(new DiscreteTokenDatasetPlanReadinessGate(
                                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults()
                                        .withHighPaddingRateThreshold(0.33d)
                                        .withKnownSolutionWarnings(false),
                                true))
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                                .experimentName("gram-nqueens")
                                .runId("run-001")
                                .seed(2026L)
                                .minimumCheckpointStep(12L)
                                .build());

        DiscreteTokenDatasetCheckpointResumePolicy parsed =
                DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(policy.toMetadata());

        assertEquals(policy, parsed);
        assertEquals(policy.toMetadata(), parsed.toMetadata());
        assertTrue(parsed.expectation().active());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT, parsed.compatibilityMode());
        assertEquals(0.33d, parsed.currentPlanGate().diagnosticsPolicy().highPaddingRateThreshold());
    }

    @Test
    void policyMetadataCanOmitExpectation() {
        Map<String, Object> metadata = Map.of(
                "currentPlanGate",
                DiscreteTokenDatasetPlanReadinessGate.lenient().toMetadata());

        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(metadata);

        assertEquals(DiscreteTokenDatasetPlanReadinessGate.lenient(), policy.currentPlanGate());
        assertEquals(DiscreteTokenDatasetCheckpointResumeExpectation.none(), policy.expectation());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT, policy.compatibilityMode());
    }

    @Test
    void supportsLenientAndCustomGateVariants() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.training())).build();
        DiscreteTokenDatasetCheckpointResumePolicy customPolicy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withCurrentPlanGate(DiscreteTokenDatasetPlanReadinessGate.lenient());

        assertTrue(DiscreteTokenDatasetCheckpointResumePolicy.lenient().evaluate(manifest, plan).ready());
        assertEquals(
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE,
                DiscreteTokenDatasetCheckpointResumePolicy.lenient().compatibilityMode());
        assertTrue(customPolicy.evaluate(manifest.toMetadata(), plan).ready());
    }

    @Test
    void compatibleModeAcceptsWarningBlockedCurrentReportWithWarning() {
        DiscreteTokenDatasetPlan plan = warningHeavyPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.training())).build();
        DiscreteTokenDatasetPlanReport strictReport = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());

        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.compatible().evaluate(
                        DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest),
                        strictReport);

        assertTrue(report.ready());
        assertFalse(report.currentPlanAccepted());
        assertEquals("warning-blocked", report.currentPlanGateStatus());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE, report.compatibilityMode());
        assertTrue(report.compatibilityWarnings().stream()
                .anyMatch(warning -> warning.contains("warning-blocked")));
        assertEquals("compatible-resume-warning", report.explanation().findings().get(1).code());
    }

    @Test
    void forceModeAcceptsMismatchedDatasetWithCompatibilityWarnings() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.force();

        DiscreteTokenDatasetCheckpointResumeReport report = policy.evaluate(manifest, changedPlan());

        assertTrue(report.ready());
        assertFalse(report.fingerprintMatched());
        assertTrue(report.forceAccepted());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE, report.compatibilityMode());
        assertTrue(report.rejectionReasons().isEmpty());
        assertTrue(report.compatibilityWarnings().stream()
                .anyMatch(warning -> warning.contains("fingerprint mismatch")));
        assertEquals("force", report.toMetadata().get("compatibilityMode"));
        assertEquals(true, report.toMetadata().get("forceAccepted"));
        assertEquals("force-resume-override", report.explanation().findings().get(1).code());
        policy.requireReady(manifest, changedPlan());
    }

    @Test
    void rejectsMalformedPolicyInputs() {
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.training();
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(plan.report()).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);

        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumePolicy(null, DiscreteTokenDatasetCheckpointResumeExpectation.none()));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetCheckpointResumePolicy(DiscreteTokenDatasetPlanReadinessGate.training(), null));
        assertThrows(NullPointerException.class, () -> policy.withCurrentPlanGate(null));
        assertThrows(NullPointerException.class, () -> policy.withExpectation(null));
        assertThrows(NullPointerException.class, () -> policy.withCompatibilityMode(null));
        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(
                        Map.of("currentPlanGate", "bad")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumePolicy.fromMetadata(
                        Map.of(
                                "currentPlanGate",
                                DiscreteTokenDatasetPlanReadinessGate.training().toMetadata(),
                                "expectation",
                                "bad")));
        assertThrows(NullPointerException.class, () -> policy.currentReport(null));
        assertThrows(NullPointerException.class, () -> policy.evaluate((Map<?, ?>) null, plan));
        assertThrows(NullPointerException.class, () -> policy.evaluate(manifest, null));
        assertThrows(NullPointerException.class, () -> policy.evaluate((DiscreteTokenDatasetCheckpointManifest) null, plan));
        assertThrows(NullPointerException.class, () -> policy.evaluate(snapshot, (DiscreteTokenDatasetPlan) null));
        assertThrows(NullPointerException.class, () -> policy.evaluate((DiscreteTokenDatasetCheckpointManifestSnapshot) null, plan));
        assertThrows(NullPointerException.class, () -> snapshot.resumeReport(plan, (DiscreteTokenDatasetCheckpointResumePolicy) null));
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
