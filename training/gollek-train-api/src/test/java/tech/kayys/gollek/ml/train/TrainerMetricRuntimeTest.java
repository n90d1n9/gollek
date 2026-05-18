package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class TrainerMetricRuntimeTest {

    @Test
    void updatesResetsAndSnapshotsPhaseMetrics() {
        CountingMetric trainMetric = new CountingMetric("train_count");
        CountingMetric validationMetric = new CountingMetric("validation_count");
        TrainerMetricRuntime runtime = new TrainerMetricRuntime(
                List.of(trainMetric),
                List.of(validationMetric),
                new RecordingFailures());

        GradTensor predictions = GradTensor.of(new float[] {1f}, 1);
        GradTensor targets = GradTensor.of(new float[] {1f}, 1);
        runtime.updateTrain(predictions, targets);
        runtime.updateValidation(predictions, targets);

        TrainerMetricRuntime.MetricSnapshot trainSnapshot =
                runtime.snapshotTrain(null, new ThroughputSnapshot(1L, 1L, 1L, 1L, 1L));
        TrainerMetricRuntime.MetricSnapshot validationSnapshot =
                runtime.snapshotValidation(null, new ThroughputSnapshot(1L, 1L, 1L, 1L, 1L));

        assertEquals(1.0, trainSnapshot.values().get("train_count"));
        assertEquals(Map.of("updates", 1), trainSnapshot.details().get("train_count"));
        assertEquals(1.0, validationSnapshot.values().get("validation_count"));
        assertEquals(Map.of("updates", 1), validationSnapshot.details().get("validation_count"));

        runtime.reset();

        TrainerMetricRuntime.MetricSnapshot resetSnapshot =
                runtime.snapshotTrain(null, new ThroughputSnapshot(1L, 1L, 1L, 1L, 1L));
        assertEquals(0.0, resetSnapshot.values().get("train_count"));
        assertEquals(Map.of("updates", 0), resetSnapshot.details().get("train_count"));
    }

    @Test
    void nonFiniteMetricIsAllowedOnlyBeforePhaseRecordsSamples() {
        NonFiniteMetric metric = new NonFiniteMetric();
        RecordingFailures failures = new RecordingFailures();
        TrainerMetricRuntime runtime = new TrainerMetricRuntime(
                List.of(metric),
                List.of(),
                failures);

        TrainerMetricRuntime.MetricSnapshot emptySnapshot =
                runtime.snapshotTrain(null, new ThroughputSnapshot(0L, 0L, 0L, 0L, 0L));

        assertTrue(Double.isNaN(emptySnapshot.values().get("nan_metric")));
        assertFalse(failures.invalidValueRecorded);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                runtime.snapshotTrain(null, new ThroughputSnapshot(1L, 1L, 1L, 1L, 1L)));

        assertEquals("train metric nan_metric must be finite, got NaN", error.getMessage());
        assertTrue(failures.invalidValueRecorded);
    }

    private static final class CountingMetric implements DetailedTrainingMetric {
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

        @Override
        public Map<String, Object> details() {
            return Map.of("updates", updates);
        }
    }

    private static final class NonFiniteMetric implements TrainingMetric {
        @Override
        public String name() {
            return "nan_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return Double.NaN;
        }
    }

    private static final class RecordingFailures implements TrainerMetricSnapshots.FailureRecorder {
        private boolean invalidValueRecorded;

        @Override
        public String invalidValue(
                String phase,
                String metricName,
                double value,
                boolean optimizerStepSkipped) {
            invalidValueRecorded = true;
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
