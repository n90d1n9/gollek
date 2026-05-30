package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetTrainerCheckpointRestorePlanTest {
    @TempDir
    Path rootDir;

    @Test
    void createsRestorePlanFromSelectedResumeReadyCheckpoint() throws IOException {
        writeReadyCheckpoint(rootDir.resolve("ready"), "run-ready", 100L);
        writeManifestOnlyCheckpoint(rootDir.resolve("newer-manifest-only"), "run-newer", 300L);

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady();
        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(rootDir, policy);

        DiscreteTokenDatasetTrainerCheckpointRestorePlan plan = report.requireRestorePlan();

        assertEquals(rootDir, plan.rootDir());
        assertEquals("run-ready", plan.runId());
        assertEquals("gram-nqueens", plan.experimentName());
        assertEquals("gram", plan.modelFamily());
        assertEquals(2026L, plan.seed());
        assertEquals(12L, plan.checkpointStep());
        assertTrue(plan.ready());
        assertTrue(plan.resumeReportRequired());
        assertTrue(plan.resumeReportPresent());
        assertTrue(plan.resumeReady());
        assertEquals(plan.summary(), DiscreteTokenDatasetTrainerCheckpointBridge
                .requireRestorePlan(rootDir, policy)
                .summary());
        assertEquals(plan.summary(), DiscreteTokenDatasetTrainerCheckpointInventory
                .scan(rootDir)
                .requireRestorePlan(policy)
                .summary());

        Map<String, Object> metadata = plan.toMetadata();
        assertEquals("run-ready", metadata.get("runId"));
        assertEquals(true, metadata.get("resumeReportRequired"));
        assertTrue(metadata.containsKey("paths"));
        assertTrue(metadata.containsKey("identity"));
        assertTrue(metadata.containsKey("lineage"));
        assertEquals("root", ((Map<?, ?>) metadata.get("lineage")).get("relation"));
        assertTrue(metadata.containsKey("checkpoint"));
        assertTrue(metadata.containsKey("resume"));
        assertTrue(plan.resumeMetadata().orElseThrow().containsKey("fingerprintMatch"));
        assertEquals("ready", plan.checkpointMetadata().get("status"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));

        Map<String, Object> reportMetadata = report.toMetadata();
        assertTrue(reportMetadata.containsKey("restorePlan"));
    }

    @Test
    void allowsManifestOnlyRestorePlanWhenPolicyDoesNotRequireResumeReport() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 300L);

        DiscreteTokenDatasetTrainerCheckpointRestorePlan plan =
                DiscreteTokenDatasetTrainerCheckpointBridge.requireRestorePlan(
                        rootDir,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady());

        assertEquals("run-manifest", plan.runId());
        assertEquals("manifest-only", plan.status());
        assertTrue(plan.ready());
        assertFalse(plan.resumeReportRequired());
        assertFalse(plan.resumeReportPresent());
        assertFalse(plan.resumeReady());
        assertEquals(plan.checkpointDir().toString(), plan.pathsMetadata().get("checkpointDir"));
        assertEquals("run-manifest", plan.identityMetadata().get("runId"));
        assertThrows(IllegalStateException.class, plan::requireResumeReport);
    }

    @Test
    void rejectsRestorePlanWhenNoCheckpointMatchesSelection() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 300L);

        DiscreteTokenDatasetTrainerCheckpointInspectionReport report =
                DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(
                        rootDir,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady());

        assertFalse(report.selectedRestorePlan().isPresent());
        assertThrows(IllegalStateException.class, report::requireRestorePlan);
        assertThrows(IllegalStateException.class, () -> DiscreteTokenDatasetTrainerCheckpointBridge
                .requireRestorePlan(
                        rootDir,
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady()));
    }

    @Test
    void rejectsMismatchedPlanPaths() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "run-manifest", 300L);
        DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint =
                DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(rootDir.resolve("manifest-only"));

        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointRestorePlan(
                        rootDir,
                        rootDir.resolve("other"),
                        checkpoint.manifestPath(),
                        checkpoint.resumeReportPath(),
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady(),
                        checkpoint));
    }

    private static void writeReadyCheckpoint(Path checkpointDir, String runId, long createdAt) throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(checkpointDir, plan, policy);
    }

    private static void writeManifestOnlyCheckpoint(Path checkpointDir, String runId, long createdAt)
            throws IOException {
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(
                checkpointDir,
                manifest(cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt)
                        .build());
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report,
            String runId,
            long createdAt) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName("gram-nqueens")
                .runId(runId)
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(createdAt)
                .createdBy("restore-plan-test");
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
                        DiscreteTokenDatasetSplitMode.STRATIFIED_SHUFFLED_FRACTIONS,
                        0L,
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
