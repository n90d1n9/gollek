package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TrainerSummaryReferenceMetadataTest {

    @Test
    void publishesOptionalPathsAndSkipsDisabledGradScalerPath() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryReferenceMetadata.put(
                metadata,
                new TrainerSummaryReferenceMetadata.Paths(
                        Path.of("model.safetensors"),
                        Path.of("model.properties"),
                        Path.of("best.safetensors"),
                        Path.of("optimizer.json"),
                        Path.of("scheduler.json"),
                        Path.of("grad-scaler.json"),
                        false,
                        Path.of("history.csv"),
                        Path.of("report.json"),
                        Path.of("runtime.json"),
                        Path.of("manifest.properties")),
                emptyErrors());

        assertEquals("model.safetensors", metadata.get("modelCheckpointFile"));
        assertEquals("model.properties", metadata.get("modelCheckpointMetadataFile"));
        assertEquals("best.safetensors", metadata.get("bestModelCheckpointFile"));
        assertEquals("optimizer.json", metadata.get("optimizerCheckpointFile"));
        assertEquals("scheduler.json", metadata.get("schedulerCheckpointFile"));
        assertFalse(metadata.containsKey("gradScalerCheckpointFile"));
        assertEquals("history.csv", metadata.get("trainingHistoryFile"));
        assertEquals("report.json", metadata.get("trainingReportFile"));
        assertEquals("runtime.json", metadata.get("runtimeCheckpointFile"));
        assertEquals("manifest.properties", metadata.get("checkpointManifestFile"));
    }

    @Test
    void publishesGradScalerPathWhenEnabledAndOnlyPresentErrors() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerSummaryReferenceMetadata.put(
                metadata,
                new TrainerSummaryReferenceMetadata.Paths(
                        null,
                        null,
                        null,
                        null,
                        null,
                        Path.of("grad-scaler.json"),
                        true,
                        null,
                        null,
                        null,
                        null),
                new TrainerSummaryReferenceMetadata.Errors(
                        "model load",
                        null,
                        "metadata load",
                        null,
                        "best save",
                        "best load",
                        null,
                        "optimizer save",
                        "scheduler load",
                        null,
                        "grad load",
                        "grad save",
                        "history load",
                        null,
                        "report save",
                        "manifest load",
                        null,
                        "runtime load"));

        assertEquals("grad-scaler.json", metadata.get("gradScalerCheckpointFile"));
        assertEquals("model load", metadata.get("modelCheckpointLoadError"));
        assertFalse(metadata.containsKey("modelCheckpointSaveError"));
        assertEquals("metadata load", metadata.get("modelCheckpointMetadataLoadError"));
        assertEquals("best save", metadata.get("bestModelCheckpointSaveError"));
        assertEquals("best load", metadata.get("bestModelCheckpointLoadError"));
        assertFalse(metadata.containsKey("optimizerCheckpointLoadError"));
        assertEquals("optimizer save", metadata.get("optimizerCheckpointSaveError"));
        assertEquals("scheduler load", metadata.get("schedulerCheckpointLoadError"));
        assertEquals("grad load", metadata.get("gradScalerCheckpointLoadError"));
        assertEquals("grad save", metadata.get("gradScalerCheckpointSaveError"));
        assertEquals("history load", metadata.get("trainingHistoryLoadError"));
        assertFalse(metadata.containsKey("trainingHistorySaveError"));
        assertEquals("report save", metadata.get("trainingReportSaveError"));
        assertEquals("manifest load", metadata.get("checkpointManifestLoadError"));
        assertFalse(metadata.containsKey("checkpointManifestSaveError"));
        assertEquals("runtime load", metadata.get("runtimeCheckpointLoadError"));
    }

    private static TrainerSummaryReferenceMetadata.Errors emptyErrors() {
        return new TrainerSummaryReferenceMetadata.Errors(
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null);
    }
}
