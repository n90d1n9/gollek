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

class DiscreteTokenDatasetTrainerProvenanceSpecTest {
    @TempDir
    Path checkpointDir;

    @Test
    void buildsManifestPolicyAndListenerFromOneSpec() {
        DiscreteTokenDatasetPlan plan = cleanPlan();

        DiscreteTokenDatasetTrainerProvenanceSpec spec = specBuilder(plan)
                .strict()
                .exactResumeExpectation()
                .build();

        assertEquals(checkpointDir, spec.checkpointDir());
        assertEquals("gram-nqueens", spec.manifest().experimentName());
        assertEquals("run-001", spec.manifest().runId());
        assertEquals("gram", spec.manifest().modelFamily());
        assertEquals(2026L, spec.manifest().seed());
        assertEquals(12L, spec.manifest().checkpointStep());
        assertTrue(spec.report().accepted());
        assertTrue(spec.resumeExpectation().active());
        assertEquals(DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(checkpointDir), spec.manifestPath());
        assertEquals(
                DiscreteTokenDatasetTrainerCheckpointBridge.resumeReportPath(checkpointDir),
                spec.resumeReportPath());
        assertEquals(checkpointDir.getParent(), spec.checkpointRootDir());
        assertTrue(spec.restoreSelectionPolicy().requireResumeReport());
        assertTrue(spec.restoreSelectionPolicy().expectation().active());
        assertEquals(spec.manifest(), spec.listener().manifest());
    }

    @Test
    void writesManifestAndEvaluatesResumeThroughSpec() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec spec = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();

        spec.writeManifest();
        DiscreteTokenDatasetCheckpointResumeReport resumeReport = spec.requireResumeReadyAndWriteReport();

        assertTrue(Files.isRegularFile(spec.manifestPath()));
        assertTrue(Files.isRegularFile(spec.resumeReportPath()));
        assertTrue(resumeReport.ready());
        assertEquals(spec.manifest().fingerprint().value(), spec.readSnapshot().fingerprint().value());
        assertEquals(true, spec.readResumeReportMetadata().get("ready"));
        DiscreteTokenDatasetCheckpointResumeReportSnapshot persistedResume = spec.readResumeReportSnapshot();
        assertEquals(resumeReport.status(), persistedResume.status());
        assertEquals(resumeReport.fingerprintMatch(), persistedResume.fingerprintMatch());
        persistedResume.requireReady();
        assertTrue(spec.readCheckpointSnapshot().ready());
        DiscreteTokenDatasetTrainerCheckpointRestorePlan restorePlan = spec.requireRestorePlan();
        assertEquals("run-001", restorePlan.runId());
        assertEquals(spec.checkpointRootDir(), restorePlan.rootDir());
        assertTrue(restorePlan.resumeReportRequired());
        assertTrue(spec.selectRestorePlan().isPresent());
        assertEquals(1, spec.scanCheckpointInventory().checkpointCount());
        assertTrue(spec.inspectRestoreCandidates().selectionSatisfied());
        DiscreteTokenDatasetTrainerCheckpointRestorePreflight restorePreflight =
                spec.requireRestorePreflightReady();
        assertTrue(restorePreflight.ready());
        assertEquals("ready", restorePreflight.status());
        assertEquals("run-001", restorePreflight.restorePlan().runId());
        assertEquals("ready", ((Map<?, ?>) spec.trainingReportMetadata(resumeReport).get("resume")).get("status"));
    }

    @Test
    void listenerCreatedBySpecPreflightsAndEnrichesSummary() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec spec = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();
        spec.writeManifest();
        DiscreteTokenDatasetTrainerProvenanceListener listener = spec.listener();

        listener.onTrainingStart(session());
        listener.onTrainingEnd(session(), summary());

        assertTrue(listener.lastResumeReport().orElseThrow().ready());
        TrainingSummary enriched = listener.lastEnrichedSummary().orElseThrow();
        Map<?, ?> dataset = (Map<?, ?>) enriched.metadata().get(
                DiscreteTokenDatasetTrainerCheckpointBridge.TRAINING_REPORT_METADATA_KEY);
        assertEquals("kept", enriched.metadata().get("base"));
        assertEquals("run-001", ((Map<?, ?>) dataset.get("checkpoint")).get("runId"));
        assertEquals(true, ((Map<?, ?>) dataset.get("resume")).get("ready"));
    }

    @Test
    void canRecordBlockedResumeWithoutFailingFromSpecListener() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec original = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();
        DiscreteTokenDatasetTrainerProvenanceSpec changed = specBuilder(changedPlan())
                .strict()
                .failOnResumeRejection(false)
                .build();
        original.writeManifest();
        DiscreteTokenDatasetTrainerProvenanceListener listener = changed.listener();

        listener.onTrainingStart(session());

        assertFalse(listener.lastResumeReport().orElseThrow().ready());
        assertTrue(Files.isRegularFile(changed.resumeReportPath()));
    }

    @Test
    void restorePreflightBlocksChangedCurrentPlanAfterSelection() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec original = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();
        original.writeManifest();
        original.requireResumeReadyAndWriteReport();

        DiscreteTokenDatasetTrainerProvenanceSpec changed = specBuilder(changedPlan())
                .strict()
                .exactResumeExpectation()
                .build();

        DiscreteTokenDatasetTrainerCheckpointRestorePlan selected = changed.requireRestorePlan();
        DiscreteTokenDatasetTrainerCheckpointRestorePreflight preflight = changed.evaluateRestorePreflight();

        assertEquals("run-001", selected.runId());
        assertFalse(preflight.ready());
        assertEquals("blocked", preflight.status());
        assertTrue(preflight.rejectionReasons().stream()
                .anyMatch(reason -> reason.contains("fingerprint")));
        assertEquals(
                "dataset-fingerprint-mismatch",
                ((Map<?, ?>) preflight.explanation().toMetadata()).get("primaryCode"));
        assertEquals(
                "dataset-fingerprint-mismatch",
                ((Map<?, ?>) preflight.toMetadata().get("explanation")).get("primaryCode"));
        assertTrue(((Map<?, ?>) preflight.toMetadata().get("currentPlanReport")).containsKey("readiness"));
        assertThrows(IllegalStateException.class, preflight::requireReady);
        assertEquals(false, preflight.toMetadata().get("ready"));
        assertThrows(IllegalStateException.class, changed::requireRestorePreflightReady);
    }

    @Test
    void metadataExposesOneObjectSetup() {
        DiscreteTokenDatasetTrainerProvenanceSpec spec = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .writeManifestOnStart(false)
                .writeResumeReportOnStart(false)
                .build();

        Map<String, Object> metadata = spec.toMetadata();

        assertEquals(checkpointDir.toString(), metadata.get("checkpointDir"));
        assertEquals(false, metadata.get("writeManifestOnStart"));
        assertEquals(false, metadata.get("writeResumeReportOnStart"));
        assertEquals("run-001", ((Map<?, ?>) metadata.get("manifest")).get("runId"));
        assertEquals("root", ((Map<?, ?>) metadata.get("lineage")).get("relation"));
        assertEquals(
                true,
                ((Map<?, ?>) ((Map<?, ?>) metadata.get("resumePolicy")).get("expectation")).get("active"));
        assertEquals(
                true,
                ((Map<?, ?>) ((Map<?, ?>) metadata.get("restoreSelectionPolicy")).get("expectation")).get("active"));
        assertEquals(true, metadata.get("restorePreflightAvailable"));
    }

    @Test
    void presetResumeGateFollowsMainGateUntilExplicitlyOverridden() {
        DiscreteTokenDatasetPlanReadinessGate lenient = DiscreteTokenDatasetPlanReadinessGate.lenient();
        DiscreteTokenDatasetPlanReadinessGate strict = DiscreteTokenDatasetPlanReadinessGate.strict();

        DiscreteTokenDatasetTrainerProvenanceSpec following = specBuilder(cleanPlan())
                .strict()
                .gate(lenient)
                .build();
        DiscreteTokenDatasetTrainerProvenanceSpec overridden = specBuilder(cleanPlan())
                .lenient()
                .resumeGate(strict)
                .build();

        assertEquals(lenient, following.currentPlanGate());
        assertEquals(strict, overridden.currentPlanGate());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT, following.resumeCompatibilityMode());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.COMPATIBLE,
                specBuilder(cleanPlan()).lenient().build().resumeCompatibilityMode());
    }

    @Test
    void forceResumeCompatibilityAllowsExplicitUnsafeResumePreflight() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec original = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();
        original.writeManifest();
        original.requireResumeReadyAndWriteReport();

        DiscreteTokenDatasetTrainerProvenanceSpec changed = specBuilder(changedPlan())
                .strict()
                .exactResumeExpectation()
                .forceResume()
                .build();

        DiscreteTokenDatasetTrainerCheckpointRestorePreflight preflight = changed.evaluateRestorePreflight();

        assertTrue(preflight.ready());
        assertTrue(preflight.resumeReport().forceAccepted());
        assertEquals(DiscreteTokenDatasetCheckpointResumeCompatibilityMode.FORCE, changed.resumeCompatibilityMode());
        assertTrue(preflight.resumeReport().compatibilityWarnings().stream()
                .anyMatch(warning -> warning.contains("fingerprint mismatch")));
    }

    @Test
    void builderCanAttachParentCheckpointLineage() throws IOException {
        DiscreteTokenDatasetTrainerProvenanceSpec parent = specBuilder(cleanPlan())
                .strict()
                .build();
        parent.writeManifest();

        DiscreteTokenDatasetTrainerProvenanceSpec child = specBuilder(cleanPlan())
                .strict()
                .runId("run-child")
                .checkpointStep(24L)
                .lineageFrom(parent.readSnapshot(), Map.of("source", "resume-test"))
                .build();

        assertEquals("run-001", child.manifest().lineage().originRunId());
        assertEquals("run-001", child.manifest().lineage().parentRunId());
        assertEquals(12L, child.manifest().lineage().parentCheckpointStep());
        assertEquals(1, child.manifest().lineage().generation());
        assertEquals("resume-test", child.manifest().lineage().attributes().get("source"));
    }

    @Test
    void restoreSelectionPolicyPreservesBasePolicyTogglesAndSpecExpectation() {
        DiscreteTokenDatasetTrainerProvenanceSpec spec = specBuilder(cleanPlan())
                .strict()
                .exactResumeExpectation()
                .build();

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                spec.restoreSelectionPolicy(
                        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestReady());

        assertTrue(policy.requireReady());
        assertFalse(policy.requireResumeReport());
        assertTrue(policy.failOnInventoryFailures());
        assertEquals("run-001", policy.expectation().runId());
    }

    @Test
    void rejectsMalformedBuilderInputs() {
        DiscreteTokenDatasetPlan plan = cleanPlan();

        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(null, plan));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan).gate(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan).resumeGate(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan).resumeCompatibilityMode(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan).lineage(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan).resumeExpectation(null));
        assertThrows(NullPointerException.class, () ->
                DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan)
                        .build()
                        .restoreSelectionPolicy(null));
    }

    private DiscreteTokenDatasetTrainerProvenanceSpec.Builder specBuilder(DiscreteTokenDatasetPlan plan) {
        return DiscreteTokenDatasetTrainerProvenanceSpec.builder(checkpointDir, plan)
                .experimentName("gram-nqueens")
                .runId("run-001")
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(12L)
                .createdAtEpochMillis(1_900_000_000_000L)
                .createdBy("trainer-test")
                .attributes(Map.of("suite", "spec"));
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
                return DiscreteTokenDatasetTrainerProvenanceSpecTest.summary();
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
