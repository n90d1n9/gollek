package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerMixedPrecisionMetadataTest {

    @TempDir
    Path tempDir;

    @Test
    void publishesMixedPrecisionAndGradScalerCheckpointState() throws Exception {
        Path checkpoint = Files.writeString(tempDir.resolve("grad-scaler.json"), "{}");
        Map<String, Object> metadata = new HashMap<>();

        TrainerMixedPrecisionMetadata.put(
                metadata,
                true,
                16.0,
                false,
                2,
                Map.of("scale", 16.0, "growthTracker", 3),
                new TrainerMixedPrecisionMetadata.GradScalerCheckpoint(
                        true,
                        true,
                        true,
                        checkpoint,
                        false,
                        true,
                        true,
                        null,
                        null,
                        false));

        assertEquals(Boolean.TRUE, metadata.get("mixedPrecisionEnabled"));
        assertEquals(16.0, metadata.get("mixedPrecisionLossScale"));
        assertEquals(Boolean.FALSE, metadata.get("mixedPrecisionOverflowDetected"));
        assertEquals(2, metadata.get("mixedPrecisionOverflowSkipCount"));
        assertEquals(Map.of("scale", 16.0, "growthTracker", 3), metadata.get("mixedPrecisionGradScalerState"));
        assertEquals(16.0, metadata.get("mixedPrecisionGradScalerState.scale"));
        assertEquals(3, metadata.get("mixedPrecisionGradScalerState.growthTracker"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointEnabled"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointSupported"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointResumeRequested"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointPresent"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointMissingOnResume"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointLoaded"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointSaved"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointLoadFailed"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointSaveFailed"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointFallbackUsed"));
    }

    @Test
    void publishesDisabledCheckpointFallbackState() {
        Map<String, Object> metadata = new HashMap<>();

        TrainerMixedPrecisionMetadata.put(
                metadata,
                false,
                Double.NaN,
                true,
                1,
                null,
                new TrainerMixedPrecisionMetadata.GradScalerCheckpoint(
                        false,
                        false,
                        false,
                        null,
                        true,
                        false,
                        false,
                        "load failed",
                        "save failed",
                        true));

        assertEquals(Boolean.FALSE, metadata.get("mixedPrecisionEnabled"));
        assertEquals(Double.NaN, metadata.get("mixedPrecisionLossScale"));
        assertEquals(Boolean.TRUE, metadata.get("mixedPrecisionOverflowDetected"));
        assertEquals(1, metadata.get("mixedPrecisionOverflowSkipCount"));
        assertEquals(Map.of(), metadata.get("mixedPrecisionGradScalerState"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointEnabled"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointSupported"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointResumeRequested"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointPresent"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointMissingOnResume"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointLoaded"));
        assertEquals(Boolean.FALSE, metadata.get("gradScalerCheckpointSaved"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointLoadFailed"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointSaveFailed"));
        assertEquals(Boolean.TRUE, metadata.get("gradScalerCheckpointFallbackUsed"));
    }
}
