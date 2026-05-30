package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiscreteTokenDatasetCheckpointManifestTest {
    @Test
    void buildsCheckpointManifestForAcceptedDatasetPlan() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());

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

        assertEquals(DiscreteTokenDatasetCheckpointManifest.SCHEMA_VERSION, manifest.schemaVersion());
        assertEquals("gram-nqueens", manifest.experimentName());
        assertEquals("run-001", manifest.runId());
        assertEquals("gram", manifest.modelFamily());
        assertEquals(2026L, manifest.seed());
        assertEquals(12L, manifest.checkpointStep());
        assertEquals(1_900_000_000_000L, manifest.createdAtEpochMillis());
        assertEquals("trainer-test", manifest.createdBy());
        assertEquals(report.fingerprint(), manifest.fingerprint());
        assertTrue(manifest.datasetAccepted());
        assertEquals("accepted", manifest.datasetGateStatus());
        assertEquals("dataset plan accepted", manifest.datasetPlanReport().readiness().summary());
        assertEquals("checkpoint run-001 step 12 dataset "
                + report.fingerprint().shortValue()
                + " accepted", manifest.summary());
        assertTrue(manifest.lineage().root());
        assertEquals("run-001", manifest.lineage().originRunId());
        manifest.requireDatasetAccepted();
    }

    @Test
    void exportsCheckpointMetadataAndSupportsExistingFingerprintSectionHelpers() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdAtEpochMillis(100L)
                        .build();

        Map<String, Object> metadata = manifest.toMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointManifest.SCHEMA_VERSION, metadata.get("schemaVersion"));
        assertEquals(report.fingerprint().value(), ((Map<?, ?>) metadata.get("fingerprint")).get("value"));
        assertEquals(true, metadata.get("datasetAccepted"));
        assertEquals("accepted", metadata.get("datasetGateStatus"));
        assertEquals("root", ((Map<?, ?>) metadata.get("lineage")).get("relation"));
        assertEquals(
                report.fingerprint().value(),
                ((Map<?, ?>) ((Map<?, ?>) metadata.get(
                        DiscreteTokenDatasetCheckpointManifest.DATASET_PLAN_REPORT_METADATA_KEY))
                        .get("fingerprint")).get("value"));
        assertEquals("matched", cleanPlan().verifyFingerprintMetadataSection(metadata).status());
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> manifest.attributes().put("bad", "value"));
    }

    @Test
    void verifiesCurrentPlansAndReportsAgainstManifestFingerprint() {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(
                                original.report(DiscreteTokenDatasetPlanReadinessGate.strict()))
                        .createdAtEpochMillis(100L)
                        .build();

        assertTrue(manifest.verifyPlan(original).matches());
        assertTrue(manifest.verifyReport(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).matches());
        assertFalse(manifest.verifyPlan(changedPlan()).matches());
        assertFalse(manifest.verifyReport(changedPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).matches());
    }

    @Test
    void strictManifestCanCarryRejectedDatasetReport() {
        DiscreteTokenDatasetPlanReport report = warningHeavyPlan()
                .report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdAtEpochMillis(100L)
                        .build();

        assertFalse(manifest.datasetAccepted());
        assertEquals("warning-blocked", manifest.datasetGateStatus());
        assertThrows(IllegalStateException.class, manifest::requireDatasetAccepted);
        assertTrue(manifest.summary().contains("warning-blocked"));
    }

    @Test
    void builderDefaultsUseDatasetFingerprintAsRunId() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report();

        DiscreteTokenDatasetCheckpointManifest manifest =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdAtEpochMillis(100L)
                        .build();

        assertEquals("default", manifest.experimentName());
        assertEquals(report.fingerprint().shortValue(), manifest.runId());
        assertEquals("recursive-reasoning", manifest.modelFamily());
        assertEquals("gollek", manifest.createdBy());
        assertEquals(0L, manifest.seed());
        assertEquals(0L, manifest.checkpointStep());
        assertEquals(report.fingerprint().shortValue(), manifest.lineage().originRunId());
    }

    @Test
    void carriesCheckpointLineageFromParentSnapshot() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest parent =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .runId("parent-run")
                        .checkpointStep(12L)
                        .createdAtEpochMillis(100L)
                        .build();
        DiscreteTokenDatasetCheckpointManifestSnapshot parentSnapshot =
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromManifest(parent);

        DiscreteTokenDatasetCheckpointManifest child =
                DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .runId("child-run")
                        .checkpointStep(18L)
                        .lineageFrom(parentSnapshot, Map.of("reason", "resume"))
                        .createdAtEpochMillis(200L)
                        .build();

        assertFalse(child.lineage().root());
        assertEquals("parent-run", child.lineage().originRunId());
        assertEquals("parent-run", child.lineage().parentRunId());
        assertEquals(12L, child.lineage().parentCheckpointStep());
        assertEquals(report.fingerprint().value(), child.lineage().parentDatasetFingerprint());
        assertEquals(1, child.lineage().generation());
        assertEquals("resume", child.lineage().attributes().get("reason"));
        assertEquals(child.lineage().toMetadata(), ((Map<?, ?>) child.toMetadata().get("lineage")));
    }

    @Test
    void rejectsMalformedManifestValues() {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report();

        assertThrows(NullPointerException.class, () -> DiscreteTokenDatasetCheckpointManifest.builder(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .schemaVersion(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .experimentName(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .runId(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .modelFamily(" ")
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .checkpointStep(-1L)
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdAtEpochMillis(-1L)
                        .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .createdBy(" ")
                        .build());
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .lineage(null)
                        .build());
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetCheckpointManifest.builder(report)
                        .attributes(mapWithNullValue())
                        .build());
    }

    private static Map<String, Object> mapWithNullValue() {
        java.util.HashMap<String, Object> values = new java.util.HashMap<>();
        values.put("bad", null);
        return values;
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
