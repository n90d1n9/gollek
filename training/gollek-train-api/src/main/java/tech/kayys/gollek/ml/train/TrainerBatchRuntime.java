package tech.kayys.gollek.ml.train;

import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.autograd.NoGrad;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.train.data.DataLoader.Batch;

/**
 * Runs canonical train/validation batches and records metric/throughput side effects.
 */
final class TrainerBatchRuntime {
    private final NNModule model;
    private final TrainingLossFunction lossFunction;
    private final GradScaler gradScaler;
    private final TrainerMetricRuntime metricRuntime;
    private final TrainerThroughputStats throughputStats;
    private final TrainerBatchGuards.FailureRecorder failures;
    private final TrainerRuntimeProfiler profiler;

    TrainerBatchRuntime(
            NNModule model,
            TrainingLossFunction lossFunction,
            GradScaler gradScaler,
            TrainerMetricRuntime metricRuntime,
            TrainerThroughputStats throughputStats,
            TrainerBatchGuards.FailureRecorder failures) {
        this(model, lossFunction, gradScaler, metricRuntime, throughputStats, failures, new TrainerRuntimeProfiler());
    }

    TrainerBatchRuntime(
            NNModule model,
            TrainingLossFunction lossFunction,
            GradScaler gradScaler,
            TrainerMetricRuntime metricRuntime,
            TrainerThroughputStats throughputStats,
            TrainerBatchGuards.FailureRecorder failures,
            TrainerRuntimeProfiler profiler) {
        this.model = Objects.requireNonNull(model, "model must not be null");
        this.lossFunction = Objects.requireNonNull(lossFunction, "lossFunction must not be null");
        this.gradScaler = gradScaler;
        this.metricRuntime = Objects.requireNonNull(metricRuntime, "metricRuntime must not be null");
        this.throughputStats = Objects.requireNonNull(throughputStats, "throughputStats must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.profiler = Objects.requireNonNull(profiler, "profiler must not be null");
    }

    double train(Object rawBatch, boolean zeroGradBeforeBackward) {
        Batch batch = profiler.time(TrainerRuntimeProfiler.Phase.TRAIN_BATCH_ADAPT, () -> toBatch(rawBatch, "train"));
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.train();
        try {
            profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_VALIDATE_BATCH,
                    () -> TrainerBatchGuards.requireFiniteBatchTensors(batch, "train", failures));
            if (zeroGradBeforeBackward) {
                profiler.time(TrainerRuntimeProfiler.Phase.TRAIN_ZERO_GRAD, model::zeroGrad);
            }

            GradTensor prediction = profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_FORWARD,
                    () -> Objects.requireNonNull(
                            model.forward(batch.inputs()),
                            "model forward returned null"));
            profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_VALIDATE_PREDICTION,
                    () -> TrainerBatchGuards.requireFiniteTensor(prediction, "train", "prediction", true, failures));
            GradTensor lossTensor = profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_LOSS,
                    () -> Objects.requireNonNull(
                            lossFunction.compute(prediction, batch.labels()),
                            "loss function returned null"));
            double loss = profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_VALIDATE_LOSS,
                    () -> TrainerBatchGuards.requireUsableLoss(lossTensor, "train", failures));

            profiler.time(
                    TrainerRuntimeProfiler.Phase.TRAIN_METRICS,
                    () -> metricRuntime.updateTrain(prediction, batch.labels()));
            GradTensor backwardLoss = gradScaler == null ? lossTensor : gradScaler.scale(lossTensor);
            profiler.time(TrainerRuntimeProfiler.Phase.TRAIN_BACKWARD, () -> backwardLoss.backward());
            recordThroughput(batch, true, System.nanoTime() - startedAt);
            counted = true;
            return loss;
        } finally {
            if (!counted) {
                recordThroughput(batch, true, System.nanoTime() - startedAt);
            }
        }
    }

    double validation(Object rawBatch) {
        Batch batch = profiler.time(
                TrainerRuntimeProfiler.Phase.VALIDATION_BATCH_ADAPT,
                () -> toBatch(rawBatch, "validation"));
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.eval();
        try {
            profiler.time(
                    TrainerRuntimeProfiler.Phase.VALIDATION_VALIDATE_BATCH,
                    () -> TrainerBatchGuards.requireFiniteBatchTensors(batch, "validation", failures));
            try (NoGrad ignored = NoGrad.enter()) {
                GradTensor prediction = profiler.time(
                        TrainerRuntimeProfiler.Phase.VALIDATION_FORWARD,
                        () -> Objects.requireNonNull(
                                model.forward(batch.inputs()),
                                "model forward returned null"));
                profiler.time(
                        TrainerRuntimeProfiler.Phase.VALIDATION_VALIDATE_PREDICTION,
                        () -> TrainerBatchGuards.requireFiniteTensor(
                                prediction,
                                "validation",
                                "prediction",
                                false,
                                failures));
                GradTensor lossTensor = profiler.time(
                        TrainerRuntimeProfiler.Phase.VALIDATION_LOSS,
                        () -> Objects.requireNonNull(
                                lossFunction.compute(prediction, batch.labels()),
                                "loss function returned null"));
                double loss = profiler.time(
                        TrainerRuntimeProfiler.Phase.VALIDATION_VALIDATE_LOSS,
                        () -> TrainerBatchGuards.requireUsableLoss(lossTensor, "validation", failures));
                profiler.time(
                        TrainerRuntimeProfiler.Phase.VALIDATION_METRICS,
                        () -> metricRuntime.updateValidation(prediction, batch.labels()));
                recordThroughput(batch, false, System.nanoTime() - startedAt);
                counted = true;
                return loss;
            }
        } finally {
            if (!counted) {
                recordThroughput(batch, false, System.nanoTime() - startedAt);
            }
        }
    }

    private void recordThroughput(Batch batch, boolean trainPhase, long elapsedNanos) {
        profiler.time(
                trainPhase
                        ? TrainerRuntimeProfiler.Phase.TRAIN_THROUGHPUT
                        : TrainerRuntimeProfiler.Phase.VALIDATION_THROUGHPUT,
                () -> throughputStats.record(batch, trainPhase, elapsedNanos));
    }

    private Batch toBatch(Object rawBatch, String phase) {
        return TrainerBatchGuards.toBatch(rawBatch, phase, failures);
    }

    TrainerRuntimeProfiler profiler() {
        return profiler;
    }
}
