package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

@SuppressWarnings("deprecation")
class CanonicalTrainerMetricFactoriesTest {

    @Test
    void nestedMetricsFacadeInheritsLegacyFactories() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.meanAbsoluteError().get();

        metric.update(
                GradTensor.of(new float[] {1.0f, 4.0f}, 2),
                GradTensor.of(new float[] {2.0f, 1.0f}, 2));

        assertEquals("mae", metric.name());
        assertEquals(2.0, metric.value(), 1e-6);
    }

    @Test
    void nestedMetricsFacadePreservesDetailedMetricCompatibility() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.binaryConfusionMatrix().get();

        metric.update(
                GradTensor.of(new float[] {2.0f, -2.0f, 2.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals("binary_confusion_matrix", detailedMetric.details().get("type"));
        assertEquals(1L, detailedMetric.details().get("truePositive"));
        assertEquals(1L, detailedMetric.details().get("trueNegative"));
        assertEquals(1L, detailedMetric.details().get("falsePositive"));
        assertEquals(1L, detailedMetric.details().get("falseNegative"));
    }
}
