package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetTrainerCheckpointInventoryTest {
    @TempDir
    Path rootDir;

    @Test
    void scansCheckpointRootAndCountsStates() throws IOException {
        Path readyDir = rootDir.resolve("ready");
        Path manifestOnlyDir = rootDir.resolve("manifest-only");
        Path blockedDir = rootDir.resolve("blocked");
        Path corruptDir = rootDir.resolve("corrupt");
        Files.createDirectories(rootDir.resolve("ignored"));
        writeReadyCheckpoint(readyDir, "run-ready", 200L);
        writeManifestOnlyCheckpoint(manifestOnlyDir, "run-manifest", 300L);
        writeBlockedCheckpoint(blockedDir, "run-blocked", 100L);
        Files.createDirectories(corruptDir);
        Files.writeString(
                DiscreteTokenDatasetTrainerCheckpointBridge.manifestPath(corruptDir),
                "{bad-json");

        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        assertEquals(3, inventory.checkpointCount());
        assertEquals(1, inventory.failureCount());
        assertEquals(2, inventory.readyCount());
        assertEquals(1, inventory.blockedCount());
        assertEquals(1, inventory.manifestOnlyCount());
        assertEquals(2, inventory.resumeReportCount());
        assertEquals("run-manifest", inventory.latestCheckpoint().orElseThrow().manifest().runId());
        assertEquals("run-manifest", inventory.requireLatestReadyCheckpoint().manifest().runId());
        assertEquals(1, inventory.blockedCheckpoints().size());
        assertEquals(1, inventory.manifestOnlyCheckpoints().size());
        assertTrue(inventory.failures().get(0).summary().contains("corrupt"));
        assertTrue(inventory.summary().contains("3 checkpoint(s)"));
        assertThrows(IllegalStateException.class, inventory::requireNoFailures);
    }

    @Test
    void scansSingleCheckpointDirectory() throws IOException {
        writeManifestOnlyCheckpoint(rootDir, "run-root", 100L);

        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointBridge.scanCheckpointInventory(rootDir);

        assertEquals(1, inventory.checkpointCount());
        assertEquals(0, inventory.failureCount());
        assertEquals("manifest-only", inventory.checkpoints().get(0).status());
        inventory.requireNoFailures();
    }

    @Test
    void returnsEmptyInventoryForDirectoryWithoutCheckpoints() throws IOException {
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        assertEquals(0, inventory.checkpointCount());
        assertEquals(0, inventory.failureCount());
        assertTrue(inventory.latestCheckpoint().isEmpty());
        assertThrows(IllegalStateException.class, inventory::requireLatestReadyCheckpoint);
    }

    @Test
    void rejectsMissingRootAndMalformedConstructorInputs() {
        assertThrows(
                NoSuchFileException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir.resolve("missing")));
        assertThrows(
                NullPointerException.class,
                () -> DiscreteTokenDatasetTrainerCheckpointInventory.scan(null));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInventory(null, List.of(), List.of()));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInventory(rootDir, List.of((DiscreteTokenDatasetTrainerCheckpointSnapshot) null), List.of()));
        assertThrows(
                NullPointerException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInventory.ReadFailure(null, "IOException", "bad"));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DiscreteTokenDatasetTrainerCheckpointInventory.ReadFailure(rootDir, " ", "bad"));
    }

    @Test
    void exportsImmutableMetadata() throws IOException {
        writeReadyCheckpoint(rootDir.resolve("ready"), "run-ready", 200L);
        DiscreteTokenDatasetTrainerCheckpointInventory inventory =
                DiscreteTokenDatasetTrainerCheckpointInventory.scan(rootDir);

        Map<String, Object> metadata = inventory.toMetadata();

        assertEquals(rootDir.toString(), metadata.get("rootDir"));
        assertEquals(1, metadata.get("checkpointCount"));
        assertEquals(1, ((List<?>) metadata.get("checkpoints")).size());
        assertTrue(((List<?>) metadata.get("failures")).isEmpty());
        assertThrows(UnsupportedOperationException.class, () -> metadata.put("bad", "value"));
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

    private static void writeBlockedCheckpoint(Path checkpointDir, String runId, long createdAt) throws IOException {
        DiscreteTokenDatasetPlan original = cleanPlan();
        DiscreteTokenDatasetCheckpointManifest manifest =
                manifest(original.report(DiscreteTokenDatasetPlanReadinessGate.strict()), runId, createdAt).build();
        DiscreteTokenDatasetTrainerCheckpointBridge.writeManifest(checkpointDir, manifest);
        DiscreteTokenDatasetTrainerCheckpointBridge.writeResumeReport(
                checkpointDir,
                DiscreteTokenDatasetCheckpointResumePolicy.strict().evaluate(manifest, changedPlan()));
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
