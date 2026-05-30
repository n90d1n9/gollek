package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointResumeReportSnapshotTest {
    @Test
    void rehydratesReadyResumeReportSnapshot() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetCheckpointResumeReport report = policy.evaluate(manifest, plan);

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertEquals("ready", snapshot.status());
        assertTrue(snapshot.policyTracked());
        assertTrue(snapshot.currentPlanChecked());
        assertEquals("run-001", snapshot.runId());
        assertEquals(12L, snapshot.checkpointStep());
        assertEquals(report.summary(), snapshot.summary());
        assertEquals(report.toMetadata(), snapshot.toMetadata());
        assertEquals(policy.toMetadata(), snapshot.policyMetadata());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT, snapshot.compatibilityMode());
        assertFalse(snapshot.forceAccepted());
        assertEquals(report.currentPlanReport().toMetadata(), snapshot.currentPlanReportMetadata());
        snapshot.requireReady();
    }

    @Test
    void rehydratesBlockedResumeReportSnapshot() {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changed);

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(report.toMetadata());

        assertFalse(snapshot.ready());
        assertEquals("blocked", snapshot.status());
        assertFalse(snapshot.fingerprintMatched());
        assertTrue(snapshot.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("dataset fingerprint mismatched")));
        assertThrows(IllegalStateException.class, snapshot::requireReady);
    }

    @Test
    void rehydratesForceAcceptedResumeReportSnapshot() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.force().evaluate(manifest, changedPlan());

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertTrue(snapshot.forceAccepted());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE, snapshot.compatibilityMode());
        assertFalse(snapshot.fingerprintMatched());
        assertTrue(snapshot.compatibilityWarnings().stream()
                .anyMatch(warning -> warning.contains("fingerprint mismatch")));
        snapshot.requireReady();
    }

    @Test
    void rehydratesReportWithoutCurrentPlanCheck() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);
        DiscreteTokenDatasetCheckpointResumeReport report =
                new DiscreteTokenDatasetCheckpointResumeReport(
                        checkpoint,
                        checkpoint.verifyReport(manifest.datasetPlanReport()));

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertTrue(snapshot.ready());
        assertFalse(snapshot.currentPlanChecked());
        assertTrue(snapshot.currentPlanAccepted());
        assertEquals("not-checked", snapshot.currentPlanGateStatus());
        assertFalse(snapshot.toMetadata().containsKey("currentPlanReport"));
    }

    @Test
    void exposesImmutableViews() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());

        DiscreteTokenDatasetCheckpointResumeReportSnapshot snapshot =
                DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(report);

        assertThrows(UnsupportedOperationException.class, () -> snapshot.policyMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> snapshot.currentPlanReportMetadata().put("bad", "value"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.rejectionReasons().add("bad"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.compatibilityWarnings().add("bad"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.toMetadata().put("bad", "value"));
    }

    @Test
    void rejectsMalformedMetadata() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, cleanPlan());
        Map<String, Object> metadata = report.toMetadata();

        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromReport(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "status", "blocked")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "policyTracked", false)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        removing(metadata, "currentPlanReport")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointResumeReportSnapshot.fromMetadata(
                        replacing(metadata, "rejectionReasons", List.of(7))));
    }

    private static Map<String, Object> replacing(Map<String, Object> metadata, String key, Object value) {
        Map<String, Object> copy = new LinkedHashMap<>(metadata);
        copy.put(key, value);
        return copy;
    }

    private static Map<String, Object> removing(Map<String, Object> metadata, String key) {
        Map<String, Object> copy = new LinkedHashMap<>(metadata);
        copy.remove(key);
        return copy;
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

    private static DiscreteTokenDatasetExample knownExample(String taskId, int index, int inputLength) {
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
}
