package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerBestModelCheckpointMetadataTest {

    @TempDir
    Path tempDir;

    @Test
    void publishesEnabledBestModelCheckpointState() throws Exception {
        Path checkpoint = tempDir.resolve("best.safetensors");
        Files.writeString(checkpoint, "weights");
        Map<String, Object> metadata = new HashMap<>();

        TrainerBestModelCheckpointMetadata.put(
                metadata,
                true,
                true,
                checkpoint,
                "validationMetric.accuracy",
                CanonicalTrainer.BestModelMonitorMode.MAX,
                new TrainerBestModelCheckpointMetadata.State(true, true, 4, 0.35, 0.91),
                null,
                null);

        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointRestoreRequested"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointSaved"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointRestored"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointPresent"));
        assertEquals("validationMetric.accuracy", metadata.get("bestModelCheckpointMonitor"));
        assertEquals("MAX", metadata.get("bestModelCheckpointMonitorMode"));
        assertEquals(4, metadata.get("bestModelCheckpointEpoch"));
        assertEquals(0.35, metadata.get("bestModelCheckpointValidationLoss"));
        assertEquals(0.91, metadata.get("bestModelCheckpointMonitorValue"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointSaveFailed"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointLoadFailed"));
    }

    @Test
    void publishesDisabledFailureStateWithoutCheckpointFile() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerBestModelCheckpointMetadata.put(
                metadata,
                false,
                false,
                null,
                "validation_loss",
                CanonicalTrainer.BestModelMonitorMode.MIN,
                new TrainerBestModelCheckpointMetadata.State(false, false, -1, Double.NaN, Double.NaN),
                "write failed",
                "read failed");

        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointRestoreRequested"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointSaved"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointRestored"));
        assertEquals(Boolean.FALSE, metadata.get("bestModelCheckpointPresent"));
        assertEquals("validation_loss", metadata.get("bestModelCheckpointMonitor"));
        assertEquals("MIN", metadata.get("bestModelCheckpointMonitorMode"));
        assertEquals(-1, metadata.get("bestModelCheckpointEpoch"));
        assertEquals(Double.NaN, metadata.get("bestModelCheckpointValidationLoss"));
        assertEquals(Double.NaN, metadata.get("bestModelCheckpointMonitorValue"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointSaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("bestModelCheckpointLoadFailed"));
    }
}
