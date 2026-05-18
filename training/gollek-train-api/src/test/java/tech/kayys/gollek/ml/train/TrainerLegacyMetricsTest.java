package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

@SuppressWarnings("deprecation")
class TrainerLegacyMetricsTest {

    @Test
    void adaptsTrainingMetricToCanonicalMetricAlias() {
        RecordingMetric delegate = new RecordingMetric("accuracy", 0.75);
        CanonicalTrainer.Metric metric = TrainerLegacyMetrics.metric(delegate);
        GradTensor predictions = GradTensor.of(new float[] {1.0f}, 1);
        GradTensor targets = GradTensor.of(new float[] {0.0f}, 1);

        metric.reset();
        metric.update(predictions, targets);

        assertEquals("accuracy", metric.name());
        assertEquals(0.75, metric.value());
        assertEquals(1, delegate.resetCount);
        assertSame(predictions, delegate.lastPredictions);
        assertSame(targets, delegate.lastTargets);
        assertFalse(metric instanceof CanonicalTrainer.DetailedMetric);
    }

    @Test
    void adaptsDetailedTrainingMetricToCanonicalDetailedAlias() {
        RecordingDetailedMetric delegate = new RecordingDetailedMetric();
        CanonicalTrainer.Metric metric = TrainerLegacyMetrics.metric(delegate);
        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);

        assertEquals("confusion_matrix", detailedMetric.name());
        assertEquals(1.0, detailedMetric.value());
        assertEquals(Map.of("tp", 3L), detailedMetric.details());
    }

    @Test
    void legacySupplierKeepsFactoryValidationDeferredUntilMetricCreation() {
        Supplier<CanonicalTrainer.Metric> supplier = TrainerLegacyMetrics.legacy(() ->
                new RecordingMetric("loss", 2.0));

        assertEquals("loss", supplier.get().name());

        Supplier<CanonicalTrainer.Metric> nullFactorySupplier = TrainerLegacyMetrics.legacy(null);
        NullPointerException nullFactoryError = assertThrows(
                NullPointerException.class,
                nullFactorySupplier::get);
        assertEquals("metric factory must not be null", nullFactoryError.getMessage());
    }

    @Test
    void rejectsNullMetricFactoryResult() {
        Supplier<CanonicalTrainer.Metric> supplier = TrainerLegacyMetrics.legacy(() -> null);

        NullPointerException error = assertThrows(NullPointerException.class, supplier::get);
        assertEquals("metric factory returned null", error.getMessage());
    }

    private static class RecordingMetric implements TrainingMetric {
        private final String name;
        private final double value;
        private int resetCount;
        private GradTensor lastPredictions;
        private GradTensor lastTargets;

        private RecordingMetric(String name, double value) {
            this.name = name;
            this.value = value;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void reset() {
            resetCount++;
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            lastPredictions = predictions;
            lastTargets = targets;
        }

        @Override
        public double value() {
            return value;
        }
    }

    private static final class RecordingDetailedMetric extends RecordingMetric implements DetailedTrainingMetric {
        private RecordingDetailedMetric() {
            super("confusion_matrix", 1.0);
        }

        @Override
        public Map<String, Object> details() {
            return Map.of("tp", 3L);
        }
    }
}
