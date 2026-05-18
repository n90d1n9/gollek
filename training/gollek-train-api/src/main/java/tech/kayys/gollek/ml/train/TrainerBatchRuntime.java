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

    TrainerBatchRuntime(
            NNModule model,
            TrainingLossFunction lossFunction,
            GradScaler gradScaler,
            TrainerMetricRuntime metricRuntime,
            TrainerThroughputStats throughputStats,
            TrainerBatchGuards.FailureRecorder failures) {
        this.model = Objects.requireNonNull(model, "model must not be null");
        this.lossFunction = Objects.requireNonNull(lossFunction, "lossFunction must not be null");
        this.gradScaler = gradScaler;
        this.metricRuntime = Objects.requireNonNull(metricRuntime, "metricRuntime must not be null");
        this.throughputStats = Objects.requireNonNull(throughputStats, "throughputStats must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
    }

    double train(Object rawBatch, boolean zeroGradBeforeBackward) {
        Batch batch = toBatch(rawBatch, "train");
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.train();
        try {
            TrainerBatchGuards.requireFiniteBatchTensors(batch, "train", failures);
            if (zeroGradBeforeBackward) {
                model.zeroGrad();
            }

            GradTensor prediction = Objects.requireNonNull(
                    model.forward(batch.inputs()),
                    "model forward returned null");
            TrainerBatchGuards.requireFiniteTensor(prediction, "train", "prediction", true, failures);
            GradTensor lossTensor = Objects.requireNonNull(
                    lossFunction.compute(prediction, batch.labels()),
                    "loss function returned null");
            double loss = TrainerBatchGuards.requireUsableLoss(lossTensor, "train", failures);

            metricRuntime.updateTrain(prediction, batch.labels());
            GradTensor backwardLoss = gradScaler == null ? lossTensor : gradScaler.scale(lossTensor);
            backwardLoss.backward();
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
        Batch batch = toBatch(rawBatch, "validation");
        long startedAt = System.nanoTime();
        boolean counted = false;
        model.eval();
        try {
            TrainerBatchGuards.requireFiniteBatchTensors(batch, "validation", failures);
            try (NoGrad ignored = NoGrad.enter()) {
                GradTensor prediction = Objects.requireNonNull(
                        model.forward(batch.inputs()),
                        "model forward returned null");
                TrainerBatchGuards.requireFiniteTensor(prediction, "validation", "prediction", false, failures);
                GradTensor lossTensor = Objects.requireNonNull(
                        lossFunction.compute(prediction, batch.labels()),
                        "loss function returned null");
                double loss = TrainerBatchGuards.requireUsableLoss(lossTensor, "validation", failures);
                metricRuntime.updateValidation(prediction, batch.labels());
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
        throughputStats.record(batch, trainPhase, elapsedNanos);
    }

    private Batch toBatch(Object rawBatch, String phase) {
        return TrainerBatchGuards.toBatch(rawBatch, phase, failures);
    }
}
