package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

@SuppressWarnings("deprecation")
class BinaryCalibrationTrainingMetricsTest {

    @Test
    void binaryLogLossStreamsStableCrossEntropyFromLogits() {
        TrainingMetric metric = TrainingMetrics.binaryLogLoss().get();

        metric.update(
                GradTensor.of(new float[] {
                        logit(0.9),
                        logit(0.2)
                }, 2),
                GradTensor.of(new float[] {
                        1.0f,
                        0.0f
                }, 2));
        metric.update(
                GradTensor.of(new float[] {
                        logit(0.7),
                        logit(0.4)
                }, 2),
                GradTensor.of(new float[] {
                        1.0f,
                        0.0f
                }, 2));

        double expected = (-Math.log(0.9) - Math.log(0.8) - Math.log(0.7) - Math.log(0.6)) / 4.0;
        assertEquals("binary_log_loss", metric.name());
        assertEquals(expected, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_log_loss", details.get("type"));
        assertEquals(4L, details.get("samples"));
        assertEquals(2L, details.get("positives"));
        assertEquals(2L, details.get("negatives"));
        assertEquals(expected, (Double) details.get("logLoss"), 1e-6);
        assertEquals(expected, (Double) details.get("binaryCrossEntropy"), 1e-6);
        assertEquals(expected, (Double) details.get("negativeLogLikelihood"), 1e-6);
        assertEquals(Math.exp(expected), (Double) details.get("perplexity"), 1e-6);
        assertEquals(0.55, (Double) details.get("meanPositiveProbability"), 1e-6);
        assertEquals(0.5, (Double) details.get("empiricalPositiveRate"), 1e-6);
        assertEquals(0.75, (Double) details.get("meanCorrectLabelProbability"), 1e-6);
        assertEquals(-Math.log(0.6), (Double) details.get("maximumSampleLogLoss"), 1e-6);
        assertEquals("logits", details.get("input"));
        assertEquals("binary_0_1", details.get("targetEncoding"));
    }

    @Test
    void binaryCrossEntropyAliasIsAvailableThroughGollekFacade() {
        CanonicalTrainer.Metric metric = Gollek.DL.binaryCrossEntropyMetric().get();

        metric.update(
                GradTensor.of(new float[] {
                        logit(0.8),
                        logit(0.25)
                }, 2),
                GradTensor.of(new float[] {
                        1.0f,
                        0.0f
                }, 2));

        CanonicalTrainer.DetailedMetric detailed =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        double expected = (-Math.log(0.8) - Math.log(0.75)) / 2.0;
        assertEquals("binary_log_loss", metric.name());
        assertEquals(expected, metric.value(), 1e-6);
        assertEquals("binary_log_loss", detailed.details().get("type"));
        assertEquals(2L, detailed.details().get("samples"));
    }

    @Test
    void binaryCalibrationMetricsReportBrierEceAndBinDetails() {
        TrainingMetric brier = TrainingMetrics.binaryBrierScore().get();
        TrainingMetric ece = TrainingMetrics.binaryExpectedCalibrationError(5).get();
        GradTensor logits = GradTensor.of(new float[] {
                logit(0.9),
                logit(0.8),
                logit(0.35),
                logit(0.1)
        }, 4);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f,
                1.0f,
                0.0f,
                0.0f
        }, 4);

        brier.update(logits, targets);
        ece.update(logits, targets);

        assertEquals("binary_brier_score", brier.name());
        assertEquals("binary_expected_calibration_error", ece.name());
        assertEquals(0.045625, brier.value(), 1e-6);
        assertEquals(0.1875, ece.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, ece);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_calibration", details.get("type"));
        assertEquals(5, details.get("bins"));
        assertEquals(4L, details.get("samples"));
        assertEquals(2L, details.get("positives"));
        assertEquals(2L, details.get("negatives"));
        assertEquals(0.045625, (Double) details.get("brierScore"), 1e-6);
        assertEquals(0.1875, (Double) details.get("expectedCalibrationError"), 1e-6);
        assertEquals(0.35, (Double) details.get("maximumCalibrationError"), 1e-6);
        assertEquals(0.5375, (Double) details.get("meanConfidence"), 1e-6);
        assertEquals(0.5, (Double) details.get("empiricalPositiveRate"), 1e-6);
        assertEquals(List.of(1L, 1L, 0L, 0L, 2L), details.get("binCount"));
        assertDoubleList((List<?>) details.get("binConfidence"), 0.1, 0.35, null, null, 0.85);
        assertDoubleList((List<?>) details.get("binAccuracy"), 0.0, 0.0, null, null, 1.0);
        assertDoubleList((List<?>) details.get("binGap"), 0.1, 0.35, null, null, 0.15);
    }

    @Test
    void binaryExpectedCalibrationErrorRejectsInvalidBinCount() {
        assertThrows(IllegalArgumentException.class,
                () -> TrainingMetrics.binaryExpectedCalibrationError(0));
    }

    @Test
    void binaryLogLossRejectsInvalidLogits() {
        TrainingMetric metric = TrainingMetrics.binaryLogLoss().get();

        assertThrows(IllegalArgumentException.class, () -> metric.update(
                GradTensor.of(new float[] {Float.NEGATIVE_INFINITY}, 1),
                GradTensor.of(new float[] {1.0f}, 1)));
    }

    private static float logit(double probability) {
        return (float) Math.log(probability / (1.0 - probability));
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
