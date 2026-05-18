package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Reads component state dictionaries and restores trainer-owned counters.
 */
final class TrainerStateCheckpointReader {
    private TrainerStateCheckpointReader() {
    }

    static OptimizerState readOptimizer(Path checkpointFile, int fallbackStepCount)
            throws IOException, ClassNotFoundException {
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpointFile, "optimizer");
        return new OptimizerState(
                state,
                TrainerCheckpointStatePayloads.optimizerStepCount(state, fallbackStepCount));
    }

    static SchedulerState readScheduler(Path checkpointFile, int fallbackStepCount)
            throws IOException, ClassNotFoundException {
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpointFile, "scheduler");
        return new SchedulerState(
                state,
                TrainerCheckpointStatePayloads.schedulerStepCount(state, fallbackStepCount));
    }

    static GradScalerState readGradScaler(Path checkpointFile, int fallbackOverflowSkipCount)
            throws IOException, ClassNotFoundException {
        Map<String, Object> state = TrainerCheckpointIO.readMap(checkpointFile, "GradScaler");
        return new GradScalerState(
                state,
                TrainerCheckpointStatePayloads.mixedPrecisionOverflowSkipCount(
                        state,
                        fallbackOverflowSkipCount));
    }

    record OptimizerState(Map<String, Object> state, int stepCount) {
    }

    record SchedulerState(Map<String, Object> state, int stepCount) {
    }

    record GradScalerState(Map<String, Object> state, int overflowSkipCount) {
    }
}
