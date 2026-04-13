package tech.kayys.gollek.ml.metrics;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {

    @Test
    void testClassificationMetricsAccuracy() {
        // Perfect predictions
        var r1 = ClassificationMetrics.compute(
            new int[]{0, 1, 2, 0, 1},
            new int[]{0, 1, 2, 0, 1},
            3
        );
        assertEquals(1.0, r1.accuracy(), 1e-5);

        // No correct predictions
        var r2 = ClassificationMetrics.compute(
            new int[]{0, 0, 0, 0, 0},
            new int[]{1, 1, 1, 1, 1},
            3
        );
        assertEquals(0.0, r2.accuracy(), 1e-5);
    }

    @Test
    void testClassificationMetricsPrecision() {
        var r = ClassificationMetrics.compute(
            new int[]{0, 1, 1, 0, 1},
            new int[]{0, 1, 0, 0, 1},
            3
        );
        double precision = r.precision();
        assertTrue(precision > 0 && precision <= 1.0);
    }

    @Test
    void testClassificationMetricsRecall() {
        var r = ClassificationMetrics.compute(
            new int[]{0, 1, 1, 0, 1},
            new int[]{0, 1, 0, 0, 1},
            3
        );
        double recall = r.recall();
        assertTrue(recall > 0 && recall <= 1.0);
    }

    @Test
    void testClassificationMetricsF1() {
        var r = ClassificationMetrics.compute(
            new int[]{0, 1, 1, 0, 1},
            new int[]{0, 1, 0, 0, 1},
            3
        );
        double f1 = r.f1();
        assertTrue(f1 >= 0 && f1 <= 1.0);
    }

    @Test
    void testRegressionMetricsMSE() {
        double mse = tech.kayys.gollek.ml.tensor.VectorOps.sum(new float[]{0, 0, 0, 0});
        assertEquals(0.0, mse, 1e-8);
    }

    @Test
    void testRegressionMetricsRMSE() {
        double rmse = RegressionMetrics.rmse(
            new float[]{1, 2, 3, 4},
            new float[]{1, 2, 3, 4}
        );
        assertEquals(0.0, rmse, 1e-8);
    }

    @Test
    void testRegressionMetricsMAE() {
        double mae = RegressionMetrics.mae(
            new float[]{1, 2, 3},
            new float[]{2, 3, 4}
        );
        assertEquals(1.0, mae, 1e-5);
    }

    @Test
    void testAccuracyTensorInput() {
        var acc = new Accuracy();
        var preds = GradTensor.of(new float[]{0, 1, 2, 0, 1}, 5);
        var labels = GradTensor.of(new float[]{0, 1, 2, 0, 1}, 5);
        acc.update(preds.data(), labels.data());
        float result = acc.compute();
        assertEquals(1.0f, result, 1e-5f);
    }
}
