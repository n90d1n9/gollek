package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Map;

/**
 * Persists component state dictionaries with trainer-owned resume metadata.
 */
final class TrainerStateCheckpointWriter {
    private TrainerStateCheckpointWriter() {
    }

    static WriteResult writeOptimizer(
            Path checkpointFile,
            Map<String, Object> optimizerState,
            OptimizerMetadata metadata) {
        if (checkpointFile == null) {
            return WriteResult.skipped();
        }
        try {
            Map<String, Object> state = TrainerCheckpointStatePayloads.optimizer(
                    optimizerState,
                    metadata.stepCount(),
                    metadata.optimizerClass(),
                    metadata.parameterSignature());
            TrainerCheckpointIO.writeMapAtomically(checkpointFile, state);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    static WriteResult writeScheduler(
            Path checkpointFile,
            Map<String, Object> schedulerState,
            SchedulerMetadata metadata) {
        if (checkpointFile == null) {
            return WriteResult.skipped();
        }
        try {
            Map<String, Object> state = TrainerCheckpointStatePayloads.scheduler(
                    schedulerState,
                    metadata.schedulerClass(),
                    metadata.stepUnit(),
                    metadata.stepCount(),
                    metadata.optimizerParameterSignature());
            TrainerCheckpointIO.writeMapAtomically(checkpointFile, state);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    static WriteResult writeGradScaler(
            Path checkpointFile,
            Map<String, Object> gradScalerState,
            GradScalerMetadata metadata) {
        if (checkpointFile == null) {
            return WriteResult.skipped();
        }
        try {
            Map<String, Object> state = TrainerCheckpointStatePayloads.gradScaler(
                    gradScalerState,
                    metadata.gradScalerClass(),
                    metadata.overflowSkipCount());
            TrainerCheckpointIO.writeMapAtomically(checkpointFile, state);
            return WriteResult.success();
        } catch (Exception error) {
            return WriteResult.failed(error);
        }
    }

    record OptimizerMetadata(
            int stepCount,
            String optimizerClass,
            String parameterSignature) {
    }

    record SchedulerMetadata(
            String schedulerClass,
            String stepUnit,
            int stepCount,
            String optimizerParameterSignature) {
    }

    record GradScalerMetadata(
            String gradScalerClass,
            int overflowSkipCount) {
    }

    record WriteResult(boolean written, String error) {
        static WriteResult skipped() {
            return new WriteResult(false, null);
        }

        static WriteResult success() {
            return new WriteResult(true, null);
        }

        static WriteResult failed(Exception error) {
            return new WriteResult(false, error.getMessage());
        }
    }
}
