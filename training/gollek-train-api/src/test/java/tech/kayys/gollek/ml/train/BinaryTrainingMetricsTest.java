package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

class BinaryTrainingMetricsTest {

    @Test
    void binaryBestF1FindsThresholdAndReportsConfusionDetails() {
        TrainingMetric metric = TrainingMetrics.binaryBestF1Threshold().get();

        metric.update(
                GradTensor.of(new float[] {0.9f, 0.8f, 0.7f, 0.4f, 0.3f, 0.2f}, 6),
                GradTensor.of(new float[] {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}, 6));

        assertEquals("binary_best_f1", metric.name());
        assertEquals(0.75, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_threshold_optimization", details.get("type"));
        assertEquals("f1", details.get("objective"));
        assertEquals(Boolean.TRUE, details.get("defined"));
        assertEquals(6L, details.get("samples"));
        assertEquals(0.3f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(0.75, (Double) details.get("f1"), 1e-6);
        assertEquals(0.6, (Double) details.get("precision"), 1e-6);
        assertEquals(1.0, (Double) details.get("recall"), 1e-6);
        assertEquals(3L, details.get("positives"));
        assertEquals(3L, details.get("negatives"));
        assertEquals(3L, details.get("truePositive"));
        assertEquals(1L, details.get("trueNegative"));
        assertEquals(2L, details.get("falsePositive"));
        assertEquals(0L, details.get("falseNegative"));
        assertEquals(List.of(
                List.of(1L, 2L),
                List.of(0L, 3L)),
                details.get("matrix"));
    }

    @Test
    void binaryBestF1UsesDeterministicTieBreakForConservativeThreshold() {
        TrainingMetric metric = TrainingMetrics.binaryBestF1().get();

        metric.update(
                GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        assertEquals(2.0 / 3.0, metric.value(), 1e-6);
        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        assertEquals(2.0f, (Float) detailed.details().get("threshold"), 1e-6f);
        assertEquals(1.0, (Double) detailed.details().get("precision"), 1e-6);
        assertEquals(0.5, (Double) detailed.details().get("recall"), 1e-6);
    }

    @Test
    void binaryBalancedAccuracyReportsThresholdedSensitivityAndSpecificity() {
        TrainingMetric metric = TrainingMetrics.binaryBalancedAccuracy(1.0f).get();

        metric.update(
                GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        assertEquals("binary_balanced_accuracy", metric.name());
        assertEquals(0.75, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_balanced_accuracy", details.get("type"));
        assertEquals(1.0f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(0.5, (Double) details.get("recall"), 1e-6);
        assertEquals(1.0, (Double) details.get("specificity"), 1e-6);
        assertEquals(0.75, (Double) details.get("balancedAccuracy"), 1e-6);
        assertEquals(1.0 / Math.sqrt(3.0), (Double) details.get("matthewsCorrelationCoefficient"), 1e-6);
        assertEquals(0.5, (Double) details.get("cohensKappa"), 1e-6);
        assertEquals(0.75, (Double) details.get("observedAgreement"), 1e-6);
        assertEquals(0.5, (Double) details.get("expectedAgreement"), 1e-6);
        assertEquals(List.of(
                List.of(2L, 0L),
                List.of(1L, 1L)),
                details.get("matrix"));
    }

    @Test
    void binaryCohensKappaCorrectsAccuracyForChanceAgreement() {
        TrainingMetric metric = TrainingMetrics.binaryKappa(1.0f).get();

        metric.update(
                GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        assertEquals("binary_cohens_kappa", metric.name());
        assertEquals(0.5, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_cohens_kappa", details.get("type"));
        assertEquals(0.75, (Double) details.get("observedAgreement"), 1e-6);
        assertEquals(0.5, (Double) details.get("expectedAgreement"), 1e-6);
        assertEquals(0.5, (Double) details.get("cohensKappa"), 1e-6);
    }

    @Test
    void binaryMatthewsCorrelationCoefficientReportsConfusionDetails() {
        TrainingMetric metric = TrainingMetrics.binaryMcc(1.0f).get();

        metric.update(
                GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        assertEquals("binary_matthews_correlation_coefficient", metric.name());
        assertEquals(1.0 / Math.sqrt(3.0), metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_matthews_correlation_coefficient", details.get("type"));
        assertEquals(1.0f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(0.75, (Double) details.get("balancedAccuracy"), 1e-6);
        assertEquals(1.0 / Math.sqrt(3.0), (Double) details.get("matthewsCorrelationCoefficient"), 1e-6);
    }

    @Test
    void binaryPrecisionAtRecallFindsMostPreciseThresholdMeetingRecall() {
        TrainingMetric metric = TrainingMetrics.binaryPrecisionAtRecall(0.75).get();

        metric.update(
                GradTensor.of(new float[] {0.95f, 0.9f, 0.8f, 0.7f, 0.4f, 0.3f, 0.2f}, 7),
                GradTensor.of(new float[] {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, 7));

        assertEquals("binary_precision_at_recall", metric.name());
        assertEquals(0.75, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("binary_threshold_constraint", details.get("type"));
        assertEquals("precision_at_recall", details.get("objective"));
        assertEquals("minimumRecall", details.get("constraint"));
        assertEquals(0.75, (Double) details.get("constraintValue"), 1e-6);
        assertEquals(0.7f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(0.75, (Double) details.get("precision"), 1e-6);
        assertEquals(0.75, (Double) details.get("recall"), 1e-6);
        assertEquals(3L, details.get("truePositive"));
        assertEquals(2L, details.get("trueNegative"));
        assertEquals(1L, details.get("falsePositive"));
        assertEquals(1L, details.get("falseNegative"));
        assertEquals(List.of(
                List.of(2L, 1L),
                List.of(1L, 3L)),
                details.get("matrix"));
    }

    @Test
    void binaryRecallAtPrecisionFindsHighestRecallThresholdMeetingPrecision() {
        TrainingMetric metric = Gollek.DL.binaryRecallAtPrecisionMetric(0.70).get();

        metric.update(
                GradTensor.of(new float[] {0.95f, 0.9f, 0.8f, 0.7f, 0.4f, 0.3f, 0.2f}, 7),
                GradTensor.of(new float[] {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}, 7));

        assertEquals("binary_recall_at_precision", metric.name());
        assertEquals(0.75, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("recall_at_precision", details.get("objective"));
        assertEquals("minimumPrecision", details.get("constraint"));
        assertEquals(0.70, (Double) details.get("constraintValue"), 1e-6);
        assertEquals(0.7f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(0.75, (Double) details.get("precision"), 1e-6);
        assertEquals(0.75, (Double) details.get("recall"), 1e-6);
    }

    @Test
    void binaryConstrainedThresholdMetricsRejectInvalidConstraintsAndReportUndefined() {
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.binaryPrecisionAtRecall(-0.1));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.binaryPrecisionAtRecall(1.1));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.binaryRecallAtPrecision(Double.NaN));

        TrainingMetric impossible = TrainingMetrics.binaryRecallAtPrecision(0.9).get();
        impossible.update(
                GradTensor.of(new float[] {0.9f, 0.8f, 0.7f}, 3),
                GradTensor.of(new float[] {0.0f, 1.0f, 0.0f}, 3));

        assertTrue(Double.isNaN(impossible.value()));
        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, impossible);
        Map<String, Object> details = detailed.details();
        assertEquals(Boolean.FALSE, details.get("defined"));
        assertEquals(3L, details.get("samples"));
        assertEquals(1L, details.get("positives"));
        assertEquals(2L, details.get("negatives"));
    }
}
