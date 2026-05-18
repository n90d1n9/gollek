package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.trainer.api.TrainerConfig;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerMetricSnapshotsTest {

    @Test
    @SuppressWarnings("unchecked")
    void snapshotsMetricDetailsAsImmutableNestedCopies() {
        MutableDetailMetric metric = new MutableDetailMetric();

        Map<String, Object> details = TrainerMetricSnapshots.snapshotDetails(
                null,
                List.of(metric),
                "train",
                true,
                new RecordingFailures());

        Map<String, Object> metricDetails = (Map<String, Object>) details.get("detail_metric");
        assertEquals(List.of(1.0, 2.0), metricDetails.get("array"));
        assertEquals(List.of("a", "b"), metricDetails.get("labels"));

        metric.source.put("after", "mutation");
        assertFalse(metricDetails.containsKey("after"));
        assertThrows(UnsupportedOperationException.class, () -> details.put("x", Map.of()));
        assertThrows(UnsupportedOperationException.class, () -> metricDetails.put("x", 1));
        assertThrows(UnsupportedOperationException.class,
                () -> ((List<Object>) metricDetails.get("labels")).add("c"));
    }

    @Test
    void snapshotValuesStopsSessionAndRecordsDuplicateMetricNames() {
        FakeSession session = new FakeSession();
        RecordingFailures failures = new RecordingFailures();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerMetricSnapshots.snapshotValues(
                        session,
                        List.of(new ConstantMetric("same", 1.0), new ConstantMetric("same", 2.0)),
                        "train",
                        true,
                        failures));

        assertTrue(session.stopped);
        assertEquals("train metric name must be unique, duplicate: same", error.getMessage());
        assertEquals("same", failures.metricName);
    }

    @Test
    void snapshotDetailsReportsNestedNonFinitePathAndStopsSession() {
        FakeSession session = new FakeSession();
        RecordingFailures failures = new RecordingFailures();

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerMetricSnapshots.snapshotDetails(
                        session,
                        List.of(new NestedNonFiniteDetailMetric()),
                        "validation",
                        true,
                        failures));

        assertTrue(session.stopped);
        assertTrue(error.getMessage().contains("details.outer[0].bad must be finite"));
        assertEquals("validation", failures.phase);
        assertEquals("nested_detail_metric", failures.metricName);
        assertEquals("details.outer[0].bad", failures.detailPath);
    }

    private static final class MutableDetailMetric implements DetailedTrainingMetric {
        private final Map<String, Object> source = new LinkedHashMap<>();

        private MutableDetailMetric() {
            source.put("array", new double[] {1.0, 2.0});
            source.put("labels", new ArrayList<>(List.of("a", "b")));
        }

        @Override
        public String name() {
            return "detail_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 1.0;
        }

        @Override
        public Map<String, Object> details() {
            return source;
        }
    }

    private static final class NestedNonFiniteDetailMetric implements DetailedTrainingMetric {
        @Override
        public String name() {
            return "nested_detail_metric";
        }

        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 1.0;
        }

        @Override
        public Map<String, Object> details() {
            return Map.of("outer", List.of(Map.of("bad", Double.NEGATIVE_INFINITY)));
        }
    }

    private record ConstantMetric(String name, double value) implements TrainingMetric {
        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }
    }

    private static final class RecordingFailures implements TrainerMetricSnapshots.FailureRecorder {
        private String phase;
        private String metricName;
        private String detailPath;

        @Override
        public String invalidValue(
                String phase,
                String metricName,
                double value,
                boolean optimizerStepSkipped) {
            this.phase = phase;
            this.metricName = metricName;
            return phase + " metric " + metricName + " must be finite, got " + value;
        }

        @Override
        public String invalidDetail(String phase, String metricName, String detailPath, double value) {
            this.phase = phase;
            this.metricName = metricName;
            this.detailPath = detailPath;
            return phase + " metric " + metricName + " detail " + detailPath + " must be finite, got " + value;
        }

        @Override
        public String invalidName(String phase, String metricName) {
            this.phase = phase;
            this.metricName = metricName;
            return phase + " metric name must be unique, duplicate: " + metricName;
        }

        @Override
        public String invalidFailure(
                String phase,
                String metricName,
                String kind,
                String detailPath,
                RuntimeException error) {
            this.phase = phase;
            this.metricName = metricName;
            this.detailPath = detailPath;
            return phase + " metric " + metricName + " failed to produce a value: " + error.getMessage();
        }
    }

    private static final class FakeSession implements TrainerSession {
        private boolean stopped;

        @Override
        public int currentEpoch() {
            return 0;
        }

        @Override
        public int globalStep() {
            return 0;
        }

        @Override
        public TrainerConfig config() {
            return new TrainerConfig(1, 0.0, false, null);
        }

        @Override
        public TrainingSummary summary() {
            return new TrainingSummary(0, Double.NaN, -1, null, null, 0L, Map.of());
        }

        @Override
        public boolean isStopped() {
            return stopped;
        }

        @Override
        public void stop() {
            stopped = true;
        }

        @Override
        public void close() {
        }
    }
}
