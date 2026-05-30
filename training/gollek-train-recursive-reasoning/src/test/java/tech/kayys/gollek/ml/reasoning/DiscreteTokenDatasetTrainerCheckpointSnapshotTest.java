package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetTrainerCheckpointSnapshotTest {
    @TempDir
    Path checkpointDir;

    @Test
    void readsCheckpointDirectoryWithManifestAndResumeAudit() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(
                        checkpointDir,
                        plan,
                        policy);

        DiscreteTokenDatasetTrainerCheckpointSnapshot snapshot =
                DiscreteTokenDatasetTrainerCheckpointSnapshot.read(checkpointDir);

        assertTrue(snapshot.ready());
        assertTrue(snapshot.manifestReady());
        assertTrue(snapshot.resumeReportPresent());
        assertTrue(snapshot.resumeReady());
        assertEquals("ready", snapshot.status());
        assertEquals("ready", snapshot.resumeStatus());
        assertEquals(report.fingerprintMatch(), snapshot.requireResumeReady().fingerprintMatch());
        assertEquals(manifest.fingerprint(), snapshot.manifest().fingerprint());
        assertEquals(true, snapshot.toMetadata().get("ready"));
        assertEquals(true, snapshot.trainingReportMetadata().containsKey("resume"));
        snapshot.requireReady();
    }

    @Test
    void readsManifestOnlyCheckpointDirectory() throws IOException {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);

        DiscreteTokenDatasetTrainerCheckpointSnapshot snapshot =
                DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(checkpointDir);

        assertTrue(snapshot.ready());
        assertTrue(snapshot.manifestReady());
        assertFalse(snapshot.resumeReportPresent());
        assertFalse(snapshot.resumeReady());
        assertEquals("manifest-only", snapshot.status());
        assertEquals("missing", snapshot.resumeStatus());
        assertTrue(snapshot.summary().contains("resume report missing"));
        assertFalse(snapshot.toMetadata().containsKey("resume"));
        assertFalse(snapshot.trainingReportMetadata().containsKey("resume"));
        snapshot.requireReady();
        assertThrows(IllegalStateException.class, snapshot::requireResumeReport);
    }

    @Test
    void capturesBlockedResumeAudit() throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changed));

        DiscreteTokenDatasetTrainerCheckpointSnapshot snapshot =
                DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(checkpointDir);

        assertFalse(snapshot.ready());
        assertTrue(snapshot.manifestReady());
        assertTrue(snapshot.resumeReportPresent());
        assertFalse(snapshot.resumeReady());
        assertEquals("blocked", snapshot.status());
        assertEquals("blocked", snapshot.resumeStatus());
        assertThrows(IllegalStateException.class, snapshot::requireReady);
        assertThrows(IllegalStateException.class, snapshot::requireResumeReady);
    }

    @Test
    void rejectsResumeAuditForDifferentManifest() throws IOException {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifest otherManifest =
                manifest(changedPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict()))
                        .runId("run-002")
                        .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(otherManifest, changedPlan()));

        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(checkpointDir));
    }

    @Test
    void exposesImmutableMetadataViews() throws IOException {
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointSnapshot snapshot =
                new DiscreteTokenDatasetTrainerCheckpointSnapshot(
                        checkpointDir,
                        DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir),
                        DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir),
                        DiscreteTokenDatasetTrainerCheckpointBridge.readSnapshot(checkpointDir),
                        Optional.empty());

        assertThrows(UnsupportedOperationException.class, () -> snapshot.toMetadata().put("bad", "value"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> snapshot.trainingReportMetadata().put("bad", "value"));
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
