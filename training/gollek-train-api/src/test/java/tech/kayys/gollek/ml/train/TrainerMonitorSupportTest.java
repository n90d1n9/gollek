package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainerMonitorSupportTest {

    @Test
    void normalizesAndLabelsMonitorMetrics() {
        assertNull(TrainerMonitorSupport.normalizeMetric(null));
        assertNull(TrainerMonitorSupport.normalizeMetric("   "));
        assertEquals("mae", TrainerMonitorSupport.normalizeMetric(" validationMetric.mae "));
        assertEquals("accuracy", TrainerMonitorSupport.normalizeMetric(" accuracy "));

        assertEquals("validation_loss", TrainerMonitorSupport.label(null));
        assertEquals("validationMetric.mae", TrainerMonitorSupport.label("mae"));
    }

    @Test
    void rejectsBlankValidationMetricPrefixWithContextualLabel() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerMonitorSupport.normalizeMetric(
                        "validationMetric.",
                        "scheduler monitor metric"));

        assertEquals("scheduler monitor metric must not be blank", error.getMessage());
    }

    @Test
    void requiresConfiguredMetricPresentWithContextualLabel() {
        assertDoesNotThrow(() -> TrainerMonitorSupport.requireMetricPresent(
                "mae",
                List.of(new NamedMetric("mae")),
                "early stopping monitor metric"));

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                TrainerMonitorSupport.requireMetricPresent(
                        "accuracy",
                        List.of(new NamedMetric("mae")),
                        "early stopping monitor metric"));

        assertEquals("early stopping monitor metric 'accuracy' is not registered. "
                + "Add the matching metric before build().", error.getMessage());
    }

    @Test
    void monitorValueReadsValidationLossOrNamedMetric() {
        assertEquals(2.5, TrainerMonitorSupport.valueOrThrow(
                null,
                2.5,
                Map.of(),
                "Best model monitor metric"));
        assertEquals(0.25, TrainerMonitorSupport.valueOrThrow(
                "mae",
                9.0,
                Map.of("mae", 0.25),
                "Best model monitor metric"));
        assertTrue(Double.isNaN(TrainerMonitorSupport.valueOrNaN("missing", 9.0, Map.of())));

        IllegalStateException error = assertThrows(IllegalStateException.class, () ->
                TrainerMonitorSupport.valueOrThrow(
                        "missing",
                        9.0,
                        Map.of(),
                        "Best model monitor metric"));

        assertEquals("Best model monitor metric 'missing' is not available. "
                + "Add it with .metric(...) or trainingOptions().*Metric().", error.getMessage());
    }

    @Test
    void earlyStoppingSummaryUsesValidationLossFallbacksOrCustomMetricState() {
        TrainingSummary base = new TrainingSummary(
                3,
                0.8,
                2,
                1.0,
                0.9,
                123L,
                Map.of("epochsWithoutImprovement", 4));

        assertEquals(2, TrainerMonitorSupport.earlyStoppingBestEpoch(null, base, 8));
        assertEquals(0.8, TrainerMonitorSupport.earlyStoppingBestValue(null, base, 4.2));
        assertEquals(0.9, TrainerMonitorSupport.earlyStoppingLatestValue(null, base, 4.4));
        assertEquals(4, TrainerMonitorSupport.earlyStoppingEpochsWithoutImprovement(null, base, 10));

        assertEquals(8, TrainerMonitorSupport.earlyStoppingBestEpoch("mae", base, 8));
        assertEquals(4.2, TrainerMonitorSupport.earlyStoppingBestValue("mae", base, 4.2));
        assertEquals(4.4, TrainerMonitorSupport.earlyStoppingLatestValue("mae", base, 4.4));
        assertEquals(10, TrainerMonitorSupport.earlyStoppingEpochsWithoutImprovement("mae", base, 10));
    }

    private record NamedMetric(String name) implements TrainingMetric {
        @Override
        public void reset() {
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
        }

        @Override
        public double value() {
            return 0.0;
        }
    }
}
