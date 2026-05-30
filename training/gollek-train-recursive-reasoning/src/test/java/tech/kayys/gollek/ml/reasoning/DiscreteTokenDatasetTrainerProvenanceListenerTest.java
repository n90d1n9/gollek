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
import tech.kayys.gollek.trainer.api.TrainerConfig;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class DiscreteTokenDatasetTrainerProvenanceListenerTest {
    @TempDir
    Path checkpointDir;

    @Test
    void writesManifestOnTrainingStart() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.create(checkpointDir, manifest);

        listener.onTrainingStart(session());

        assertTrue(Files.isRegularFile(DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir)));
        assertTrue(listener.lastResumeReport().isEmpty());
        assertEquals(
                manifest.fingerprint().value(),
                DiscreteTokenDatasetTrainerCheckpointBridge.readSnapshot(checkpointDir).fingerprint().value());
    }

    @Test
    void preflightsExistingManifestAndEnrichesFinalSummary() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.withResumePreflight(
                        checkpointDir,
                        manifest,
                        plan,
                        policy);

        listener.onTrainingStart(session());
        listener.onTrainingEnd(session(), summary());

        assertTrue(listener.lastResumeReport().orElseThrow().ready());
        assertTrue(Files.isRegularFile(DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir)));
        TrainingSummary enriched = listener.lastEnrichedSummary().orElseThrow();
        Map<?, ?> dataset = (Map<?, ?>) enriched.metadata().get(
                DiscreteTokenDatasetTrainerCheckpointBridge.TRAINING_REPORT_METADATA_KEY);
        assertEquals("kept", enriched.metadata().get("base"));
        assertEquals(true, dataset.get("datasetAccepted"));
        assertEquals(true, ((Map<?, ?>) dataset.get("resume")).get("ready"));
        assertEquals(
                DiscreteTokenDatasetTrainerCheckpointBridge.MANIFEST_FILE_NAME,
                dataset.get("manifestFileName"));
    }

    @Test
    void blocksTrainingStartWhenExistingManifestDoesNotMatchCurrentPlan() throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest savedManifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifest currentManifest =
                manifest(changed.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, savedManifest);
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, currentManifest)
                        .currentPlan(changed)
                        .resumePolicy(DiscreteTokenDatasetCheckpointResumePolicy.strict())
                        .build();

        assertThrows(IllegalStateException.class, () -> listener.onTrainingStart(session()));
        assertFalse(listener.lastResumeReport().orElseThrow().ready());
    }

    @Test
    void canRecordBlockedResumeWithoutFailingTrainingStart() throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetPlan changed = changedPlan();
        DiscreteTokenDatasetCheckpointManifest savedManifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetCheckpointManifest currentManifest =
                manifest(changed.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, savedManifest);
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, currentManifest)
                        .currentPlan(changed)
                        .resumePolicy(DiscreteTokenDatasetCheckpointResumePolicy.strict())
                        .failOnResumeRejection(false)
                        .build();

        listener.onTrainingStart(session());

        assertFalse(listener.lastResumeReport().orElseThrow().ready());
        assertTrue(Files.isRegularFile(DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir)));
        assertEquals(
                currentManifest.fingerprint().value(),
                DiscreteTokenDatasetTrainerCheckpointBridge.readSnapshot(checkpointDir).fingerprint().value());
    }

    @Test
    void canDisableManifestAndResumeReportWrites() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, manifest)
                        .currentPlan(plan)
                        .writeManifestOnStart(false)
                        .writeResumeReportOnStart(false)
                        .build();

        listener.onTrainingStart(session());

        assertTrue(listener.lastResumeReport().orElseThrow().ready());
        assertFalse(Files.isRegularFile(DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir)));
    }

    @Test
    void exposesCurrentMetadataBeforeAndAfterResumeCheck() throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerProvenanceListener listener =
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, manifest)
                        .currentPlan(plan)
                        .build();

        assertFalse(listener.trainingReportMetadata().containsKey("resume"));
        listener.onTrainingStart(session());

        assertEquals(true, ((Map<?, ?>) listener.trainingReportMetadata().get("resume")).get("ready"));
        assertEquals(checkpointDir, listener.checkpointDir());
        assertEquals(manifest, listener.manifest());
    }

    @Test
    void rejectsMalformedBuilderInputs() {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(plan.report(DiscreteTokenDatasetPlanReadinessGate.strict())).build();

        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceListener.create(null, manifest));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceListener.create(checkpointDir, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, manifest).currentPlan(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceListener.builder(checkpointDir, manifest).resumePolicy(null));
    }

    private TrainerSession session() {
        return new TrainerSession() {
            private boolean stopped;

            @Override
            public int currentEpoch() {
                return 0;
            }

            @Override
            public int globalStep() {
                return 0;
            }

            @Override
            public TrainerConfig config() {
                return new TrainerConfig(1, 0.0d, false, checkpointDir);
            }

            @Override
            public TrainingSummary summary() {
                return DiscreteTokenDatasetTrainerProvenanceListenerTest.summary();
            }

            @Override
            public boolean isStopped() {
                return stopped;
            }

            @Override
            public void stop() {
                stopped = true;
            }

            @Override
            public void close() {}
        };
    }

    private static TrainingSummary summary() {
        return new TrainingSummary(1, 0.5d, 0, 0.75d, 0.5d, 10L, Map.of("base", "kept"));
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
