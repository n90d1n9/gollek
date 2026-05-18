package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerStateCheckpointWriterTest {
    @TempDir
    Path tempDir;

    @Test
    void writesOptimizerStateWithTrainerMetadata() throws Exception {
        Path checkpoint = tempDir.resolve("optimizer.state");
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("momentum", 0.9);

        TrainerStateCheckpointWriter.WriteResult result = TrainerStateCheckpointWriter.writeOptimizer(
                checkpoint,
                componentState,
                new TrainerStateCheckpointWriter.OptimizerMetadata(
                        7,
                        "AdamW",
                        "w:4,b:1"));

        assertTrue(result.written());
        assertNull(result.error());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "optimizer");
        assertEquals(0.9, state.get("momentum"));
        assertEquals(7, state.get("trainerOptimizerStepCount"));
        assertEquals("AdamW", state.get("trainerOptimizerClass"));
        assertEquals("w:4,b:1", state.get("trainerOptimizerParameterSignature"));
    }

    @Test
    void writesSchedulerStateWithTrainerMetadata() throws Exception {
        Path checkpoint = tempDir.resolve("scheduler.state");
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("lastLr", 0.001);

        TrainerStateCheckpointWriter.WriteResult result = TrainerStateCheckpointWriter.writeScheduler(
                checkpoint,
                componentState,
                new TrainerStateCheckpointWriter.SchedulerMetadata(
                        "CosineAnnealing",
                        "EPOCH",
                        3,
                        "w:4,b:1"));

        assertTrue(result.written());
        assertNull(result.error());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "scheduler");
        assertEquals(0.001, state.get("lastLr"));
        assertEquals("CosineAnnealing", state.get("trainerSchedulerClass"));
        assertEquals("EPOCH", state.get("trainerSchedulerStepUnit"));
        assertEquals(3, state.get("trainerSchedulerStepCount"));
        assertEquals("w:4,b:1", state.get("trainerSchedulerOptimizerParameterSignature"));
    }

    @Test
    void writesGradScalerStateWithTrainerMetadata() throws Exception {
        Path checkpoint = tempDir.resolve("grad-scaler.state");
        Map<String, Object> componentState = new LinkedHashMap<>();
        componentState.put("scale", 1024.0);

        TrainerStateCheckpointWriter.WriteResult result = TrainerStateCheckpointWriter.writeGradScaler(
                checkpoint,
                componentState,
                new TrainerStateCheckpointWriter.GradScalerMetadata(
                        "GradScaler",
                        2));

        assertTrue(result.written());
        assertNull(result.error());
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpoint, "GradScaler");
        assertEquals(1024.0, state.get("scale"));
        assertEquals("GradScaler", state.get("trainerGradScalerClass"));
        assertEquals(true, state.get("trainerMixedPrecisionEnabled"));
        assertEquals(2, state.get("trainerMixedPrecisionOverflowSkipCount"));
    }

    @Test
    void skippedWhenTargetIsAbsent() {
        TrainerStateCheckpointWriter.WriteResult result = TrainerStateCheckpointWriter.writeOptimizer(
                null,
                Map.of(),
                new TrainerStateCheckpointWriter.OptimizerMetadata(0, "SGD", "none"));

        assertFalse(result.written());
        assertNull(result.error());
    }
}
