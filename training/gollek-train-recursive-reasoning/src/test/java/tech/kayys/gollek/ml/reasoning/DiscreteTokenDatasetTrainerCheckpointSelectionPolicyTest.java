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

class DiscreteTokenDatasetTrainerCheckpointSelectionPolicyTest {
    @TempDir
    Path rootDir;

    @Test
    void selectsLatestReadyAndLatestResumeReadyCheckpoints() throws IOException {
        writeReadyCheckpoint(rootDir.resolve("ready"), "gram-nqueens", "run-ready", 100L, 12L);
        writeManifestOnlyCheckpoint(rootDir.resolve("manifest-only"), "gram-nqueens", "run-manifest", 300L, 13L);
        writeBlockedCheckpoint(rootDir.resolve("blocked"), "gram-nqueens", "run-blocked", 400L, 14L);

        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latest =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latest();
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latestReady =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady();
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy latestResumeReady =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady();

        assertEquals("run-blocked", latest.require(inventory).manifest().runId());
        assertEquals("run-manifest", latestReady.require(inventory).manifest().runId());
        assertEquals("run-ready", latestResumeReady.require(inventory).manifest().runId());
        assertEquals("run-ready", inventory.requireCheckpoint(latestResumeReady).manifest().runId());
        assertEquals("run-ready", DiscreteTokenDatasetTrainerCheckpointBridge
                .requireCheckpoint(rootDir, latestResumeReady)
                .manifest()
                .runId());
        assertEquals(2, latestReady.candidates(inventory).size());
        assertEquals(1, latestResumeReady.candidates(inventory).size());

        DiscreteTokenDatasetTrainerCheckpointSnapshot manifestOnly = latestReady.require(inventory);
        assertTrue(latestResumeReady.rejectionReasons(manifestOnly)
                .contains("checkpoint resume report is missing"));
    }

    @Test
    void strictSelectionRejectsInventoryFailures() throws IOException {
        writeManifestOnlyCheckpoint(rootDir.resolve("ready"), "gram-nqueens", "run-ready", 100L, 12L);
        Path corruptDir = rootDir.resolve("corrupt");
        Files.createDirectories(corruptDir);
        Files.writeString(
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(corruptDir),
                "{bad-json");

        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        assertEquals("run-ready", DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady()
                .require(inventory)
                .manifest()
                .runId());
        assertThrows(
                IllegalStateException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestReady()
                        .require(inventory));
    }

    @Test
    void selectionCanRejectUnhealthyLineage() throws IOException {
        DiscreteTokenDatasetPlanReport report = cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict());
        DiscreteTokenDatasetCheckpointManifest orphan = manifest(
                report,
                "gram-nqueens",
                "run-orphan",
                100L,
                12L)
                .lineage(new DiscreteTokenDatasetCheckpointLineage(
                        "run-origin",
                        "missing-parent",
                        10L,
                        report.fingerprint().value(),
                        1,
                        DiscreteTokenDatasetCheckpointLineage.RESUME_RELATION,
                        Map.of()))
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(rootDir.resolve("orphan"), orphan);
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy base =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady();
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy guarded =
                base.withFailOnLineageIssues(true);

        assertEquals("run-orphan", base.require(inventory).manifest().runId());
        assertThrows(IllegalStateException.class, () -> guarded.require(inventory));
        assertFalse(DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(inventory, guarded)
                .selectionSatisfied());
        assertTrue(DiscreteTokenDatasetTrainerCheckpointInspectionReport.inspect(inventory, guarded)
                .selectionFailure()
                .orElseThrow()
                .contains("missing-parent"));
    }

    @Test
    void filtersCandidatesWithResumeExpectation() throws IOException {
        writeReadyCheckpoint(rootDir.resolve("nqueens"), "gram-nqueens", "run-nqueens", 100L, 12L);
        writeReadyCheckpoint(rootDir.resolve("graph"), "gram-graph", "run-graph", 200L, 20L);
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        DiscreteTokenDatasetCheckpointResumeExpectation expectation =
                DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .experimentName("gram-nqueens")
                        .minimumCheckpointStep(10L)
                        .build();
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestResumeReady()
                        .withExpectation(expectation);

        assertEquals("run-nqueens", policy.require(inventory).manifest().runId());
        assertEquals(1, policy.candidates(inventory).size());
        assertTrue(policy.rejectionReasons(inventory.latestCheckpoint().orElseThrow()).stream()
                .anyMatch(reason -> reason.contains("experimentName expected gram-nqueens")));

        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy impossible =
                policy.withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                        .runId("missing-run")
                        .build());

        assertFalse(impossible.select(inventory).isPresent());
        assertThrows(IllegalStateException.class, () -> impossible.require(inventory));
    }

    @Test
    void exportsMetadataAndRoundTrips() {
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy policy =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.strictLatestResumeReady()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.builder()
                                .modelFamily("gram")
                                .minimumCheckpointStep(5L)
                                .build());

        Map<String, Object> metadata = policy.toMetadata();
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy rehydrated =
                DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.fromMetadata(metadata);

        assertTrue(rehydrated.requireReady());
        assertTrue(rehydrated.requireResumeReport());
        assertTrue(rehydrated.failOnInventoryFailures());
        assertFalse(rehydrated.failOnLineageIssues());
        assertTrue(rehydrated.expectation().active());
        assertEquals(policy.summary(), rehydrated.summary());
        DiscreteTokenDatasetTrainerCheckpointSelectionPolicy lineageGuarded =
                policy.withFailOnLineageIssues(true);
        assertTrue(DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.fromMetadata(lineageGuarded.toMetadata())
                .failOnLineageIssues());
        assertEquals(true, lineageGuarded.toMetadata().get("failOnLineageIssues"));
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
    }

    @Test
    void rejectsMalformedInputs() {
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointSelectionPolicy(true, false, false, null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.builder().expectation(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady().select(null));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.latestReady().candidates(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.fromMetadata(Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.fromMetadata(Map.of(
                        "requireReady", true,
                        "requireResumeReport", true,
                        "failOnInventoryFailures", true,
                        "failOnLineageIssues", "bad",
                        "expectation", Map.of("active", false))));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointSelectionPolicy.fromMetadata(Map.of(
                        "requireReady", true,
                        "requireResumeReport", true,
                        "failOnInventoryFailures", true,
                        "expectation", "bad")));
    }

    private static void writeReadyCheckpoint(
            Path checkpointDir,
            String experimentName,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetPlan plan = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(
                plan.report(DiscreteTokenDatasetPlanReadinessGate.strict()),
                experimentName,
                runId,
                createdAt,
                checkpointStep)
                .build();
        DiscreteTokenDatasetCheckpointResumePolicy policy =
                DiscreteTokenDatasetCheckpointResumePolicy.strict()
                        .withExpectation(DiscreteTokenDatasetCheckpointResumeExpectation.exactFromManifest(manifest));
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.requireResumeReadyAndWriteReport(checkpointDir, plan, policy);
    }

    private static void writeManifestOnlyCheckpoint(
            Path checkpointDir,
            String experimentName,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(
                checkpointDir,
                manifest(
                        cleanPlan().report(DiscreteTokenDatasetPlanReadinessGate.strict()),
                        experimentName,
                        runId,
                        createdAt,
                        checkpointStep)
                        .build());
    }

    private static void writeBlockedCheckpoint(
            Path checkpointDir,
            String experimentName,
            String runId,
            long createdAt,
            long checkpointStep) throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest = manifest(
                original.report(DiscreteTokenDatasetPlanReadinessGate.strict()),
                experimentName,
                runId,
                createdAt,
                checkpointStep)
                .build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changedPlan()));
    }

    private static DiscreteTokenDatasetCheckpointManifest.Builder manifest(
            DiscreteTokenDatasetPlanReport report,
            String experimentName,
            String runId,
            long createdAt,
            long checkpointStep) {
        return DiscreteTokenDatasetCheckpointManifest.builder(report)
                .experimentName(experimentName)
                .runId(runId)
                .modelFamily("gram")
                .seed(2026L)
                .checkpointStep(checkpointStep)
                .createdAtEpochMillis(createdAt)
                .createdBy("selection-policy-test");
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
