package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointManifestSnapshotTest {
    @Test
    void rehydratesSnapshotFromPersistedManifestMetadata() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetPlanReport report = plan.report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .experimentName("gram-nqueens")
                        .runId("run-001")
                        .modelFamily("gram")
                        .seed(2026L)
                        .checkpointStep(12L)
                        .createdAtEpochMillis(1_900_000_000_000L)
                        .createdBy("trainer-test")
                        .attributes(Map.of("split", "stratified"))
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(manifest.toMetadata());

        assertEquals(DiscreteTokenDatasetCheckpointManifest.SCHEMA_VERSION, snapshot.schemaVersion());
        assertTrue(snapshot.isCurrentSchema());
        assertEquals("gram-nqueens", snapshot.experimentName());
        assertEquals("run-001", snapshot.runId());
        assertEquals("gram", snapshot.modelFamily());
        assertEquals(2026L, snapshot.seed());
        assertEquals(12L, snapshot.checkpointStep());
        assertEquals(1_900_000_000_000L, snapshot.createdAtEpochMillis());
        assertEquals("trainer-test", snapshot.createdBy());
        assertEquals(report.fingerprint(), snapshot.fingerprint());
        assertEquals(report.fingerprint(), snapshot.datasetPlanReportFingerprint());
        assertTrue(snapshot.datasetAccepted());
        assertEquals("accepted", snapshot.datasetGateStatus());
        assertEquals("root", snapshot.lineage().relation());
        assertEquals("run-001", snapshot.lineage().originRunId());
        assertEquals("stratified", snapshot.attributes().get("split"));
        assertEquals(manifest.summary(), snapshot.summary());
        assertEquals(manifest.toMetadata(), snapshot.toMetadata());
        assertTrue(snapshot.verifyPlan(plan).matches());
        assertTrue(snapshot.verifyReport(report).matches());
        snapshot.requireCurrentSchema();
        snapshot.requireDatasetAccepted();
    }

    @Test
    void rehydratesSnapshotDirectlyFromManifest() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .createdAtEpochMillis(100L)
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(manifest);

        assertEquals(manifest.fingerprint(), snapshot.fingerprint());
        assertEquals(manifest.toMetadata(), snapshot.toMetadata());
    }

    @Test
    void rehydratesSnapshotWithMissingLineageAsRootForBackwardCompatibility() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .runId("old-run")
                        .createdAtEpochMillis(100L)
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(removing(manifest.toMetadata(), "lineage"));

        assertTrue(snapshot.lineage().root());
        assertEquals("old-run", snapshot.lineage().originRunId());
        assertEquals("lineage root old-run", snapshot.lineage().summary());
    }

    @Test
    void rehydratesChildLineageFromPersistedMetadata() {
        DiscreteTokenDatasetCheckpointManifest parent =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .runId("parent-run")
                        .checkpointStep(12L)
                        .createdAtEpochMillis(100L)
                        .build();
        DiscreteTokenDatasetCheckpointManifest child =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .runId("child-run")
                        .checkpointStep(18L)
                        .lineageFrom(DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(parent))
                        .createdAtEpochMillis(200L)
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(child.toMetadata());

        assertFalse(snapshot.lineage().root());
        assertEquals("parent-run", snapshot.lineage().originRunId());
        assertEquals("parent-run", snapshot.lineage().parentRunId());
        assertEquals(1, snapshot.lineage().generation());
    }

    @Test
    void rejectsMismatchedTopLevelAndReportFingerprints() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .createdAtEpochMillis(100L)
                        .build();
        Map<String, Object> metadata = new LinkedHashMap<>(manifest.toMetadata());
        metadata.put("fingerprint", changedPlan().fingerprint().toMetadata());

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(metadata));

        assertTrue(error.getMessage().contains("fingerprint"));
    }

    @Test
    void rejectsMismatchedTopLevelAndReportReadinessStatus() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .createdAtEpochMillis(100L)
                        .build();
        Map<String, Object> metadata = new LinkedHashMap<>(manifest.toMetadata());
        metadata.put("datasetGateStatus", "warning-blocked");

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(metadata));

        assertTrue(error.getMessage().contains("datasetGateStatus"));
    }

    @Test
    void carriesRejectedDatasetStatusForResumeGuards() {
        DiscreteTokenDatasetPlanReport report =
                warningHeavyPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdAtEpochMillis(100L)
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(manifest.toMetadata());

        assertFalse(snapshot.datasetAccepted());
        assertEquals("warning-blocked", snapshot.datasetGateStatus());
        assertTrue(snapshot.summary().contains("warning-blocked"));
        assertTrue(snapshot.verifyReport(report).matches());
        assertThrows(IllegalStateException.class, snapshot::requireDatasetAccepted);
    }

    @Test
    void rejectsMalformedMetadata() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .createdAtEpochMillis(100L)
                        .build();

        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "fingerprint", "bad")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        removing(manifest.toMetadata(),
                                DiscreteTokenDatasetCheckpointManifest.DATASET_PLAN_REPORT_METADATA_KEY)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "checkpointStep", -1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "datasetAccepted", "not-bool")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "attributes", badAttributeKeyMap())));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "attributes", badAttributeValueMap())));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "lineage", "bad")));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "lineage", lineageWithBadAttributeKey())));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(
                        replacing(manifest.toMetadata(), "lineage", lineageWithBadAttributeValue())));
    }

    @Test
    void exposesImmutableMetadataViews() {
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(cleanPlan().report())
                        .createdAtEpochMillis(100L)
                        .attributes(Map.of("source", "unit-test"))
                        .build();

        DiscreteTokenDatasetCheckpointManifestSnapshot snapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(manifest.toMetadata());

        assertThrows(UnsupportedOperationException.class, () -> snapshot.attributes().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> snapshot.datasetPlanReportMetadata().put("bad", "value"));
        assertThrows(UnsupportedOperationException.class, () -> snapshot.toMetadata().put("bad", "value"));
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

    private static Map<Object, Object> badAttributeKeyMap() {
        Map<Object, Object> values = new HashMap<>();
        values.put(7, "bad");
        return values;
    }

    private static Map<String, Object> badAttributeValueMap() {
        Map<String, Object> values = new HashMap<>();
        values.put("bad", null);
        return values;
    }

    private static Map<String, Object> lineageWithBadAttributeKey() {
        Map<String, Object> lineage = new LinkedHashMap<>();
        lineage.put("originRunId", "run");
        lineage.put("generation", 0);
        lineage.put("relation", "root");
        lineage.put("attributes", badAttributeKeyMap());
        return lineage;
    }

    private static Map<String, Object> lineageWithBadAttributeValue() {
        Map<String, Object> lineage = new LinkedHashMap<>();
        lineage.put("originRunId", "run");
        lineage.put("generation", 0);
        lineage.put("relation", "root");
        lineage.put("attributes", badAttributeValueMap());
        return lineage;
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
