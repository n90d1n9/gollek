package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainerStateCheckpointReaderTest {
    @TempDir
    Path tempDir;

    @Test
    void readsOptimizerStateAndRestoresStepCount() throws Exception {
        Path checkpoint = tempDir.resolve("optimizer.state");
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("momentum", 0.9);
        state.put("trainerOptimizerStepCount", "12");
        TrainerCheckpointIO.writeMapAtomically(checkpoint, state);

        TrainerStateCheckpointReader.OptimizerState restored =
                TrainerStateCheckpointReader.readOptimizer(checkpoint, 4);

        assertEquals(0.9, restored.state().get("momentum"));
        assertEquals("12", restored.state().get("trainerOptimizerStepCount"));
        assertEquals(12, restored.stepCount());
    }

    @Test
    void readsSchedulerStateAndUsesFallbackForInvalidStepCount() throws Exception {
        Path checkpoint = tempDir.resolve("scheduler.state");
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("lastLr", 0.001);
        state.put("trainerSchedulerStepCount", "not-a-number");
        TrainerCheckpointIO.writeMapAtomically(checkpoint, state);

        TrainerStateCheckpointReader.SchedulerState restored =
                TrainerStateCheckpointReader.readScheduler(checkpoint, 8);

        assertEquals(0.001, restored.state().get("lastLr"));
        assertEquals(8, restored.stepCount());
    }

    @Test
    void readsGradScalerStateAndClampsNegativeOverflowSkipCount() throws Exception {
        Path checkpoint = tempDir.resolve("grad-scaler.state");
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("scale", 1024.0);
        state.put("trainerMixedPrecisionOverflowSkipCount", -3);
        TrainerCheckpointIO.writeMapAtomically(checkpoint, state);

        TrainerStateCheckpointReader.GradScalerState restored =
                TrainerStateCheckpointReader.readGradScaler(checkpoint, 5);

        assertEquals(1024.0, restored.state().get("scale"));
        assertEquals(0, restored.overflowSkipCount());
    }

    @Test
    void exposesMutableStateForComponentLoadersLikeCheckpointIoDid() throws Exception {
        Path checkpoint = tempDir.resolve("optimizer.state");
        TrainerCheckpointIO.writeMapAtomically(checkpoint, Map.of("weightDecay", 0.01));

        TrainerStateCheckpointReader.OptimizerState restored =
                TrainerStateCheckpointReader.readOptimizer(checkpoint, 0);
        restored.state().put("loaderScratch", true);

        assertSame(Boolean.TRUE, restored.state().get("loaderScratch"));
    }
}
