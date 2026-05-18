package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.GradScaler;
import tech.kayys.gollek.train.data.DataLoader.Batch;

class TrainerBatchRuntimeTest {

    @Test
    void trainBatchRunsBackwardMetricsAndThroughput() {
        ScaleModel model = new ScaleModel(2f);
        CountingMetric trainMetric = new CountingMetric("train_updates");
        TrainerThroughputStats throughput = new TrainerThroughputStats();
        TrainerBatchRuntime runtime = batchRuntime(
                model,
                List.of(trainMetric),
                List.of(),
                throughput,
                null,
                (predictions, targets) -> predictions.sum());

        double loss = runtime.train(batch(3f, 0f), true);

        assertEquals(6.0, loss, 1e-6);
        assertEquals(1, trainMetric.updates);
        assertEquals(3.0f, model.weight.grad().data()[0], 1e-6f);
        assertEquals(1L, throughput.trainEpoch().batchCount());
        assertEquals(1L, throughput.trainEpoch().sampleCount());
    }

    @Test
    void validationBatchRunsWithoutBackpropagating() {
        ScaleModel model = new ScaleModel(2f);
        CountingMetric validationMetric = new CountingMetric("validation_updates");
        TrainerThroughputStats throughput = new TrainerThroughputStats();
        TrainerBatchRuntime runtime = batchRuntime(
                model,
                List.of(),
                List.of(validationMetric),
                throughput,
                null,
                (predictions, targets) -> predictions.sum());

        double loss = runtime.validation(batch(3f, 0f));

        assertEquals(6.0, loss, 1e-6);
        assertEquals(1, validationMetric.updates);
        assertNull(model.weight.grad());
        assertEquals(1L, throughput.validationEpoch().batchCount());
        assertEquals(1L, throughput.validationEpoch().sampleCount());
    }

    @Test
    void trainBatchScalesBackwardLossWhenGradScalerIsEnabled() {
        ScaleModel model = new ScaleModel(2f);
        TrainerBatchRuntime runtime = batchRuntime(
                model,
                List.of(),
                List.of(),
                new TrainerThroughputStats(),
                GradScaler.builder().initScale(4.0).build(),
                (predictions, targets) -> predictions.sum());

        runtime.train(batch(3f, 0f), true);

        assertEquals(12.0f, model.weight.grad().data()[0], 1e-6f);
    }

    @Test
    void trainBatchRecordsThroughputWhenLossIsRejected() {
        ScaleModel model = new ScaleModel(2f);
        TrainerThroughputStats throughput = new TrainerThroughputStats();
        RecordingBatchFailures failures = new RecordingBatchFailures();
        TrainerBatchRuntime runtime = new TrainerBatchRuntime(
                model,
                (predictions, targets) -> GradTensor.of(new float[] {1f, 2f}, 2),
                null,
                metricRuntime(List.of(), List.of()),
                throughput,
                failures);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                runtime.train(batch(3f, 0f), true));

        assertEquals("train loss tensor must contain exactly one value before backward, got shape [2]", error.getMessage());
        assertEquals(1L, throughput.trainEpoch().batchCount());
        assertEquals(1L, throughput.trainEpoch().sampleCount());
        assertEquals(1, failures.discardCount);
    }

    private static TrainerBatchRuntime batchRuntime(
            NNModule model,
            List<TrainingMetric> trainMetrics,
            List<TrainingMetric> validationMetrics,
            TrainerThroughputStats throughput,
            GradScaler gradScaler,
            TrainingLossFunction lossFunction) {
        return new TrainerBatchRuntime(
                model,
                lossFunction,
                gradScaler,
                metricRuntime(trainMetrics, validationMetrics),
                throughput,
                new RecordingBatchFailures());
    }

    private static TrainerMetricRuntime metricRuntime(
            List<TrainingMetric> trainMetrics,
            List<TrainingMetric> validationMetrics) {
        return new TrainerMetricRuntime(
                trainMetrics,
                validationMetrics,
                new RecordingMetricFailures());
    }

    private static Batch batch(float input, float label) {
        return new Batch(
                GradTensor.of(new float[] {input}, 1),
                GradTensor.of(new float[] {label}, 1));
    }

    private static final class ScaleModel extends NNModule {
        private final Parameter weight;

        private ScaleModel(float weight) {
            this.weight = registerParameter("weight", GradTensor.of(new float[] {weight}, 1));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return input.mul(weight.data());
        }
    }

    private static final class CountingMetric implements TrainingMetric {
        private final String name;
        private int updates;

        private CountingMetric(String name) {
            this.name = name;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void reset() {
            updates = 0;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            updates++;
        }

        @Override
        public double value() {
            return updates;
        }
    }

    private static final class RecordingBatchFailures implements TrainerBatchGuards.FailureRecorder {
        private int discardCount;

        @Override
        public String invalidBatch(
                String phase,
                String reason,
                String message,
                boolean optimizerStepSkipped) {
            return message;
        }

        @Override
        public String invalidLossShape(
                String phase,
                String shape,
                long elements,
                boolean optimizerStepSkipped) {
            return phase + " loss tensor must contain exactly one value before backward, got shape " + shape;
        }

        @Override
        public String nonFinite(
                String phase,
                String kind,
                double value,
                String label,
                boolean optimizerStepSkipped) {
            return phase + " " + label + " must be finite, got " + value;
        }

        @Override
        public void discardPendingGradients() {
            discardCount++;
        }
    }

    private static final class RecordingMetricFailures implements TrainerMetricSnapshots.FailureRecorder {
        @Override
        public String invalidValue(
                String phase,
                String metricName,
                double value,
                boolean optimizerStepSkipped) {
            return phase + " metric " + metricName + " must be finite, got " + value;
        }

        @Override
        public String invalidDetail(String phase, String metricName, String detailPath, double value) {
            return phase + " metric " + metricName + " detail " + detailPath + " must be finite, got " + value;
        }

        @Override
        public String invalidName(String phase, String metricName) {
            return phase + " metric name must be unique, duplicate: " + metricName;
        }

        @Override
        public String invalidFailure(
                String phase,
                String metricName,
                String kind,
                String detailPath,
                RuntimeException error) {
            return phase + " metric " + metricName + " failed while reading " + kind;
        }
    }
}
