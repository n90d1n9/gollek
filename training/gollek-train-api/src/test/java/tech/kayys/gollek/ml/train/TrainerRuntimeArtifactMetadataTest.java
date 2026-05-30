package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerRuntimeArtifactMetadataTest {

    @TempDir
    Path tempDir;

    @Test
    void publishesRuntimeArtifactsWhenPresent() throws Exception {
        Path history = Files.writeString(tempDir.resolve("history.csv"), "epoch,loss\n");
        Path report = Files.writeString(tempDir.resolve("report.json"), "{}");
        Path manifest = Files.writeString(tempDir.resolve("checkpoint.properties"), "version=1\n");
        Path runtime = Files.writeString(tempDir.resolve("runtime.json"), "{}");
        Map<String, Object> metadata = new HashMap<>();

        TrainerRuntimeArtifactMetadata.put(
                metadata,
                new TrainerRuntimeArtifactMetadata.Artifact(true, history, false, true, true, null, null),
                new TrainerRuntimeArtifactMetadata.SaveOnlyArtifact(true, report, true, null),
                new TrainerRuntimeArtifactMetadata.Artifact(true, manifest, false, true, true, null, null),
                false,
                new TrainerRuntimeArtifactMetadata.RuntimeCheckpoint(runtime, false, false, null));

        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryPresent"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryMissingOnResume"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistorySaved"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryLoadFailed"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistorySaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportPresent"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportSaved"));
        assertEquals(Boolean.FALSE, metadata.get("trainingReportSaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestPresent"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestSaved"));
        assertEquals(Boolean.FALSE, metadata.get("checkpointManifestIntegrityMismatch"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointPresent"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointResumeAllowed"));
        assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointIntegrityMismatch"));
        assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointResumeSkipped"));
        assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointLoadFailed"));
        assertEquals("runtime-checkpoint-ready", metadata.get("runtimeCheckpointResumeDecision"));
        assertEquals(
                "continue with runtime checkpoint resume when requested",
                metadata.get("runtimeCheckpointRecommendedAction"));
        Map<String, Object> plan = runtimeCheckpointResumePlan(metadata);
        assertEquals(1, plan.get("version"));
        assertEquals(Boolean.TRUE, plan.get("present"));
        assertEquals(Boolean.TRUE, plan.get("resumeAllowed"));
        assertEquals(Boolean.FALSE, plan.get("integrityMismatch"));
        assertEquals(Boolean.FALSE, plan.get("resumeSkipped"));
        assertEquals(Boolean.FALSE, plan.get("loadFailed"));
        assertEquals("runtime-checkpoint-ready", plan.get("decision"));
        assertEquals(
                "continue with runtime checkpoint resume when requested",
                plan.get("recommendedAction"));
        assertThrows(UnsupportedOperationException.class, plan::clear);
    }

    @Test
    void publishesFailureAndMissingRuntimeArtifactState() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerRuntimeArtifactMetadata.put(
                metadata,
                new TrainerRuntimeArtifactMetadata.Artifact(true, null, true, false, false, "load failed", "save failed"),
                new TrainerRuntimeArtifactMetadata.SaveOnlyArtifact(true, null, false, "report failed"),
                new TrainerRuntimeArtifactMetadata.Artifact(
                        true,
                        null,
                        true,
                        true,
                        false,
                        "manifest load failed",
                        "manifest save failed"),
                true,
                new TrainerRuntimeArtifactMetadata.RuntimeCheckpoint(null, true, true, "runtime load failed"));

        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryPresent"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryMissingOnResume"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistoryLoaded"));
        assertEquals(Boolean.FALSE, metadata.get("trainingHistorySaved"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistoryLoadFailed"));
        assertEquals(Boolean.TRUE, metadata.get("trainingHistorySaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("trainingReportPresent"));
        assertEquals(Boolean.FALSE, metadata.get("trainingReportSaved"));
        assertEquals(Boolean.TRUE, metadata.get("trainingReportSaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestMissingOnResume"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestLoaded"));
        assertEquals(Boolean.FALSE, metadata.get("checkpointManifestSaved"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestLoadFailed"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestSaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("checkpointManifestIntegrityMismatch"));
        assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointPresent"));
        assertEquals(Boolean.FALSE, metadata.get("runtimeCheckpointResumeAllowed"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointIntegrityMismatch"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointResumeSkipped"));
        assertEquals(Boolean.TRUE, metadata.get("runtimeCheckpointLoadFailed"));
        assertEquals(
                "runtime-checkpoint-integrity-mismatch",
                metadata.get("runtimeCheckpointResumeDecision"));
        assertEquals(
                "skip runtime checkpoint and rebuild runtime state from trainer artifacts",
                metadata.get("runtimeCheckpointRecommendedAction"));
        Map<String, Object> plan = runtimeCheckpointResumePlan(metadata);
        assertEquals(Boolean.FALSE, plan.get("present"));
        assertEquals(Boolean.FALSE, plan.get("resumeAllowed"));
        assertEquals(Boolean.TRUE, plan.get("integrityMismatch"));
        assertEquals(Boolean.TRUE, plan.get("resumeSkipped"));
        assertEquals(Boolean.TRUE, plan.get("loadFailed"));
        assertEquals("runtime load failed", plan.get("loadError"));
        assertEquals(
                "runtime-checkpoint-integrity-mismatch",
                plan.get("decision"));
        assertEquals(
                "skip runtime checkpoint and rebuild runtime state from trainer artifacts",
                plan.get("recommendedAction"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> runtimeCheckpointResumePlan(Map<String, Object> metadata) {
        return (Map<String, Object>) metadata.get("runtimeCheckpointResumePlan");
    }
}
