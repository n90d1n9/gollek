package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerSummaryMetadataTest {

    @Test
    void checkpointResumeOverviewMarksPartialOnlyWhenResumeWasRequested() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryMetadata.putCheckpointResumeOverview(
                metadata,
                false,
                List.of("optimizer"),
                List.of("model: signature mismatch"));

        assertEquals(List.of("optimizer"), metadata.get("checkpointResumeMissingArtifacts"));
        assertEquals(List.of("model: signature mismatch"), metadata.get("checkpointResumeCompatibilityMismatches"));
        assertEquals(List.of("model: signature mismatch"), metadata.get("checkpointCompatibilityMismatches"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointCompatibilityMismatch"));
        assertEquals(Boolean.FALSE, metadata.get("checkpointResumePartial"));
    }

    @Test
    void checkpointResumeOverviewStoresImmutableSnapshots() {
        Map<String, Object> metadata = new HashMap<>();
        List<String> missing = new ArrayList<>(List.of("optimizer"));
        List<String> mismatches = new ArrayList<>(List.of("scheduler"));

        TrainerSummaryMetadata.putCheckpointResumeOverview(metadata, true, missing, mismatches);
        missing.add("late-missing");
        mismatches.add("late-mismatch");

        assertEquals(List.of("optimizer"), metadata.get("checkpointResumeMissingArtifacts"));
        assertEquals(List.of("scheduler"), metadata.get("checkpointResumeCompatibilityMismatches"));
        assertThrows(UnsupportedOperationException.class, () -> immutableMissingArtifacts(metadata).clear());
    }

    @Test
    void optionalPathWritesOnlyWhenPresentAndIncluded() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryMetadata.putOptionalPath(metadata, "modelCheckpointFile", Path.of("checkpoint.bin"));
        TrainerSummaryMetadata.putOptionalPath(metadata, "optimizerCheckpointFile", null);
        TrainerSummaryMetadata.putOptionalPath(metadata, "gradScalerCheckpointFile", Path.of("grad.bin"), false);

        assertEquals("checkpoint.bin", metadata.get("modelCheckpointFile"));
        assertFalse(metadata.containsKey("optimizerCheckpointFile"));
        assertFalse(metadata.containsKey("gradScalerCheckpointFile"));
    }

    @Test
    void optionalErrorWritesOnlyWhenPresent() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryMetadata.putOptionalError(metadata, "modelCheckpointLoadError", "bad-shape");
        TrainerSummaryMetadata.putOptionalError(metadata, "modelCheckpointSaveError", null);

        assertEquals("bad-shape", metadata.get("modelCheckpointLoadError"));
        assertFalse(metadata.containsKey("modelCheckpointSaveError"));
    }

    @Test
    void checkpointStatusWritesResumeAndLoadSaveFlags(@TempDir Path tempDir) throws Exception {
        Map<String, Object> metadata = new HashMap<>();
        Path checkpoint = Files.writeString(tempDir.resolve("optimizer.bin"), "state");

        TrainerSummaryMetadata.putSupportedCheckpointStatus(
                metadata,
                "optimizerCheckpoint",
                true,
                true,
                true,
                checkpoint,
                false,
                true,
                false,
                "load-failed",
                null);

        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointSupported"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointResumeRequested"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointPresent"));
        assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointMissingOnResume"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointLoaded"));
        assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointSaved"));
        assertEquals(Boolean.TRUE, metadata.get("optimizerCheckpointLoadFailed"));
        assertEquals(Boolean.FALSE, metadata.get("optimizerCheckpointSaveFailed"));
    }

    @Test
    void artifactStatusWritesCommonArtifactLifecycleFlags() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryMetadata.putArtifactStatus(
                metadata,
                "trainingHistory",
                true,
                Path.of("missing-history.csv"),
                true,
                false,
                true,
                null,
                "save-failed");

        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryPresent"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryMissingOnResume"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistorySaved"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryLoadFailed"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistorySaveFailed"));
    }

    @Test
    void saveOnlyArtifactStatusSkipsLoadFields(@TempDir Path tempDir) throws Exception {
        Map<String, Object> metadata = new HashMap<>();
        Path report = Files.writeString(tempDir.resolve("report.json"), "{}");

        TrainerSummaryMetadata.putSaveOnlyArtifactStatus(
                metadata,
                "trainingReport",
                true,
                report,
                true,
                "write-failed");

        assertEquals(Boolean.TRUE, metadata.get("trainingReportEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportPresent"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportSaved"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportSaveFailed"));
        assertFalse(metadata.containsKey("trainingReportLoaded"));
        assertFalse(metadata.containsKey("trainingReportLoadFailed"));
        assertFalse(metadata.containsKey("trainingReportMissingOnResume"));
    }

    private static List<?> immutableMissingArtifacts(Map<String, Object> metadata) {
        return (List<?>) metadata.get("checkpointResumeMissingArtifacts");
    }
}
