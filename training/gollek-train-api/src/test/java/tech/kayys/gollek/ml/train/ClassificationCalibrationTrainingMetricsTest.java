package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class ClassificationCalibrationTrainingMetricsTest {

    @Test
    void classificationCalibrationMetricsReportBrierEceAndBinDetails() {
        TrainingMetric brier = TrainingMetrics.classificationBrierScore().get();
        TrainingMetric ece = TrainingMetrics.classificationExpectedCalibrationError(5).get();
        GradTensor logits = GradTensor.of(new float[] {
                logProb(0.8), logProb(0.1), logProb(0.1),
                logProb(0.4), logProb(0.5), logProb(0.1),
                logProb(0.2), logProb(0.3), logProb(0.5)
        }, 3, 3);
        GradTensor targets = GradTensor.of(new float[] {
                0.0f,
                0.0f,
                2.0f
        }, 3);

        brier.update(logits, targets);
        ece.update(logits, targets);

        assertEquals("classification_brier_score", brier.name());
        assertEquals("classification_expected_calibration_error", ece.name());
        assertEquals(1.06 / 3.0, brier.value(), 1e-6);
        assertEquals(1.0 / 15.0, ece.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, ece);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_calibration", details.get("type"));
        assertEquals("top_label", details.get("mode"));
        assertEquals(5, details.get("bins"));
        assertEquals(3, details.get("classes"));
        assertEquals(3L, details.get("samples"));
        assertEquals(2L, details.get("correct"));
        assertEquals(2.0 / 3.0, (Double) details.get("accuracy"), 1e-6);
        assertEquals(1.06 / 3.0, (Double) details.get("brierScore"), 1e-6);
        assertEquals(1.0 / 15.0, (Double) details.get("expectedCalibrationError"), 1e-6);
        assertEquals(0.2, (Double) details.get("maximumCalibrationError"), 1e-6);
        assertEquals(0.6, (Double) details.get("meanConfidence"), 1e-6);
        assertEquals(List.of(0L, 0L, 2L, 0L, 1L), details.get("binCount"));
        assertDoubleList((List<?>) details.get("binConfidence"), null, null, 0.5, null, 0.8);
        assertDoubleList((List<?>) details.get("binAccuracy"), null, null, 0.5, null, 1.0);
        assertDoubleList((List<?>) details.get("binGap"), null, null, 0.0, null, 0.2);
    }

    @Test
    void classificationCalibrationMetricsAcceptOneHotTargets() {
        TrainingMetric brier = TrainingMetrics.classificationBrierScore().get();
        TrainingMetric ece = TrainingMetrics.classificationExpectedCalibrationError(5).get();
        GradTensor logits = GradTensor.of(new float[] {
                logProb(0.8), logProb(0.1), logProb(0.1),
                logProb(0.4), logProb(0.5), logProb(0.1),
                logProb(0.2), logProb(0.3), logProb(0.5)
        }, 3, 3);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f
        }, 3, 3);

        brier.update(logits, targets);
        ece.update(logits, targets);

        assertEquals(1.06 / 3.0, brier.value(), 1e-6);
        assertEquals(1.0 / 15.0, ece.value(), 1e-6);
    }

    @Test
    void classificationExpectedCalibrationErrorRejectsInvalidBinCount() {
        assertThrows(IllegalArgumentException.class,
                () -> TrainingMetrics.classificationExpectedCalibrationError(0));
    }

    private static float logProb(double probability) {
        return (float) Math.log(probability);
    }

    private static void assertDoubleList(List<?> actual, Double first, Double second, Double third,
            Double fourth, Double fifth) {
        Double[] expected = {first, second, third, fourth, fifth};
        assertEquals(expected.length, actual.size());
        for (int i = 0; i < expected.length; i++) {
            Double expectedValue = expected[i];
            if (expectedValue == null) {
                assertEquals(null, actual.get(i));
            } else {
                assertEquals(expectedValue, (Double) actual.get(i), 1e-6);
            }
        }
    }
}
