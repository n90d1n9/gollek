package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;

/**
 * Applies consistent save/error handling for state-dictionary checkpoints.
 */
final class TrainerStateCheckpointPersistence {
    private TrainerStateCheckpointPersistence() {
    }

    static Result persistOptimizer(
            Path checkpointFile,
            Optimizer optimizer,
            int optimizerStepCount) {
        if (checkpointFile == null || optimizer == null || !optimizer.supportsStateDict()) {
            return Result.skipped();
        }
        return fromWriteResult(TrainerStateCheckpointWriter.writeOptimizer(
                checkpointFile,
                optimizer.stateDict(),
                new TrainerStateCheckpointWriter.OptimizerMetadata(
                        optimizerStepCount,
                        optimizer.getClass().getName(),
                        TrainerMetadataSupport.parameterSignature(optimizer.parameters()))));
    }

    static Result persistScheduler(
            Path checkpointFile,
            LRScheduler scheduler,
            String schedulerStepUnit,
            int schedulerStepCount,
            Optimizer optimizer) {
        if (checkpointFile == null || scheduler == null || !scheduler.supportsStateDict()) {
            return Result.skipped();
        }
        return fromWriteResult(TrainerStateCheckpointWriter.writeScheduler(
                checkpointFile,
                scheduler.stateDict(),
                new TrainerStateCheckpointWriter.SchedulerMetadata(
                        scheduler.getClass().getName(),
                        schedulerStepUnit,
                        schedulerStepCount,
                        TrainerMetadataSupport.parameterSignature(optimizer.parameters()))));
    }

    static Result persistGradScaler(
            Path checkpointFile,
            GradScaler gradScaler,
            int mixedPrecisionOverflowSkipCount) {
        if (checkpointFile == null || gradScaler == null || !gradScaler.supportsStateDict()) {
            return Result.skipped();
        }
        return fromWriteResult(TrainerStateCheckpointWriter.writeGradScaler(
                checkpointFile,
                gradScaler.stateDict(),
                new TrainerStateCheckpointWriter.GradScalerMetadata(
                        gradScaler.getClass().getName(),
                        mixedPrecisionOverflowSkipCount)));
    }

    private static Result fromWriteResult(TrainerStateCheckpointWriter.WriteResult writeResult) {
        if (writeResult.written()) {
            return Result.savedResult();
        }
        if (writeResult.error() != null) {
            return Result.failed(writeResult.error());
        }
        return Result.skipped();
    }

    record Result(boolean stateChanged, boolean saved, String saveError) {
        static Result skipped() {
            return new Result(false, false, null);
        }

        static Result savedResult() {
            return new Result(true, true, null);
        }

        static Result failed(String saveError) {
            return new Result(true, false, saveError);
        }
    }
}
