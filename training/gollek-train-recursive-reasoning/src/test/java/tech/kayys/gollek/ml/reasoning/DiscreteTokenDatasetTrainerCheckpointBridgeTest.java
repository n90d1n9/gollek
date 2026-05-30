package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class DiscreteTokenDatasetTrainerCheckpointBridgeTest {
    @TempDir
    Path checkpointDir;

    @Test
    void writesDatasetManifestAndResumeReportBesideTrainerCheckpoints() throws IOException {
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

        assertTrue(Files.isRegularFile(
                checkpointDir.resolve(DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME)));
        assertTrue(Files.isRegularFile(
                checkpointDir.resolve(DiscreteTokenDatasetTrainerCheckpointBridge.RESUME_REPORT_FILE_NAME)));
        assertTrue(report.ready());
        assertTrue(report.policyTracked());
        assertEquals(
                manifest.summary(),
                DiscreteTokenDatasetTrainerCheckpointBridge.readSnapshot(checkpointDir).summary());
        assertEquals(
                true,
                DiscreteTokenDatasetTrainerCheckpointBridge.readResumeReportMetadata(checkpointDir).get("ready"));
        DiscreteTokenDatasetCheckpointResumeReportSnapshot resumeSnapshot =
                DiscreteTokenDatasetTrainerCheckpointBridge.readResumeReportSnapshot(checkpointDir);
        assertEquals(report.status(), resumeSnapshot.status());
        assertEquals(report.checkpoint().runId(), resumeSnapshot.runId());
        assertEquals(report.fingerprintMatch(), resumeSnapshot.fingerprintMatch());
        assertEquals(report.policyMetadata(), resumeSnapshot.policyMetadata());
        resumeSnapshot.requireReady();
        assertTrue(DiscreteTokenDatasetTrainerCheckpointBridge.readCheckpointSnapshot(checkpointDir).ready());
    }

    @Test
    void readManifestMetadataCanDriveResumePolicy() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);

        Map<String, Object> metadata =
                DiscreteTokenDatasetTrainerCheckpointBridge.readManifestMetadata(checkpointDir);
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetCheckpointResumePolicy.training().evaluate(metadata, plan);

        assertTrue(report.ready());
        assertEquals(manifest.fingerprint().value(), report.checkpoint().fingerprint().value());
    }

    @Test
    void resumePreflightBlocksChangedCurrentPlan() throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetCheckpointResumeReport report =
                DiscreteTokenDatasetTrainerCheckpointBridge.evaluateResume(
                        checkpointDir,
                        changed,
                        DiscreteTokenDatasetCheckpointResumePolicy.strict());

        assertFalse(report.ready());
        assertFalse(report.fingerprintMatched());
        assertThrows(
                IllegalStateException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReady(
                        checkpointDir,
                        changed,
                        DiscreteTokenDatasetCheckpointResumePolicy.strict()));
    }

    @Test
    void attachesRecursiveReasoningSectionToTrainingMetadata() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumePolicy.training().evaluate(manifest, plan);

        Map<String, Object> metadata =
                DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingMetadata(
                        Map.of("existing", "kept"),
                        manifest,
                        resumeReport);
        Map<?, ?> section = (Map<?, ?>) metadata.get(
                DiscreteTokenDatasetTrainerCheckpointBridge.TRAINING_REPORT_METADATA_KEY);

        assertEquals("kept", metadata.get("existing"));
        assertEquals(true, section.get("datasetAccepted"));
        assertEquals("accepted", section.get("datasetGateStatus"));
        assertEquals(
                manifest.fingerprint().value(),
                ((Map<?, ?>) section.get("fingerprint")).get("value"));
        assertEquals(true, ((Map<?, ?>) section.get("resume")).get("ready"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void attachesRecursiveReasoningSectionToTrainingSummaryMetadata() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumeReport resumeReport =
                DiscreteTokenDatasetCheckpointResumePolicy.training().evaluate(manifest, plan);
        TrainingSummary summary = new TrainingSummary(
                3,
                0.25d,
                2,
                0.5d,
                0.3d,
                120L,
                Map.of("base", "yes"));

        TrainingSummary enriched =
                DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(
                        summary,
                        manifest,
                        resumeReport);
        Map<?, ?> section = (Map<?, ?>) enriched.metadata().get(
                DiscreteTokenDatasetTrainerCheckpointBridge.TRAINING_REPORT_METADATA_KEY);

        assertEquals(3, enriched.epochCount());
        assertEquals(0.25d, enriched.bestValidationLoss());
        assertEquals("yes", enriched.metadata().get("base"));
        assertEquals(
                DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME,
                section.get("manifestFileName"));
        assertEquals(true, ((Map<?, ?>) section.get("resume")).get("ready"));
    }

    @Test
    void rejectsMalformedBridgeInputs() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        TrainingSummary summary = new TrainingSummary(1, 0.0d, 0, null, null, 0L, Map.of());

        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.evaluateResume(checkpointDir, plan, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(checkpointDir, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.trainingReportMetadata(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(null, manifest));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingSummary(summary, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerCheckpointBridge.attachToTrainingMetadata(
                        mapWithNullValue(),
                        manifest));
    }

    private static Map<String, Object> mapWithNullValue() {
        java.util.HashMap<String, Object> metadata = new java.util.HashMap<>();
        metadata.put("bad", null);
        return metadata;
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
