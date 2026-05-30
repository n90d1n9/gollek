package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class ClassificationImbalanceTrainingMetricsTest {

    @Test
    void classificationBalancedAccuracyAveragesPerClassRecallForObservedClasses() {
        TrainingMetric metric = TrainingMetrics.classificationBalancedAccuracy().get();

        metric.update(
                GradTensor.of(new float[] {
                        3.0f, 1.0f, 0.0f,
                        2.0f, 0.0f, 1.0f,
                        0.0f, 3.0f, 1.0f,
                        0.0f, 2.0f, 1.0f,
                        3.0f, 2.0f, 0.0f,
                        0.0f, 2.0f, 1.0f
                }, 6, 3),
                GradTensor.of(new float[] {
                        0.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f,
                        2.0f
                }, 6));

        assertEquals("classification_balanced_accuracy", metric.name());
        assertEquals(7.0 / 18.0, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_balanced_accuracy", details.get("type"));
        assertEquals("macro_recall_observed_classes", details.get("averaging"));
        assertEquals(3, details.get("classes"));
        assertEquals(3, details.get("definedClasses"));
        assertEquals(6L, details.get("total"));
        assertEquals(3L, details.get("correct"));
        assertEquals(0.5, (Double) details.get("accuracy"), 1e-6);
        assertEquals(7.0 / 18.0, (Double) details.get("balancedAccuracy"), 1e-6);
        assertEquals(3.0 / Math.sqrt(396.0), (Double) details.get("matthewsCorrelationCoefficient"), 1e-6);
        assertEquals(4.0 / 9.0, (Double) details.get("weightedPrecision"), 1e-6);
        assertEquals(0.5, (Double) details.get("weightedRecall"), 1e-6);
        assertEquals(7.0 / 15.0, (Double) details.get("weightedF1"), 1e-6);
        assertEquals(1.0 / 7.0, (Double) details.get("cohensKappa"), 1e-6);
        assertEquals(0.5, (Double) details.get("observedAgreement"), 1e-6);
        assertEquals(5.0 / 12.0, (Double) details.get("expectedAgreement"), 1e-6);
        assertEquals(List.of(3L, 2L, 1L), details.get("support"));
        assertEquals(List.of(3L, 3L, 0L), details.get("predictedSupport"));
        assertDoubleList((List<?>) details.get("perClassRecall"), 2.0 / 3.0, 0.5, 0.0);
        assertDoubleList((List<?>) details.get("perClassPrecision"), 2.0 / 3.0, 1.0 / 3.0, null);
        assertEquals(List.of(
                List.of(2L, 1L, 0L),
                List.of(1L, 1L, 0L),
                List.of(0L, 1L, 0L)),
                details.get("matrix"));
    }

    @Test
    void classificationMatthewsCorrelationCoefficientUsesFullConfusionMatrix() {
        TrainingMetric metric = TrainingMetrics.classificationMcc().get();

        metric.update(
                GradTensor.of(new float[] {
                        3.0f, 1.0f, 0.0f,
                        2.0f, 0.0f, 1.0f,
                        0.0f, 3.0f, 1.0f,
                        0.0f, 2.0f, 1.0f,
                        3.0f, 2.0f, 0.0f,
                        0.0f, 2.0f, 1.0f
                }, 6, 3),
                GradTensor.of(new float[] {
                        0.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f,
                        2.0f
                }, 6));

        assertEquals("classification_matthews_correlation_coefficient", metric.name());
        assertEquals(3.0 / Math.sqrt(396.0), metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_matthews_correlation_coefficient", details.get("type"));
        assertEquals(7.0 / 18.0, (Double) details.get("balancedAccuracy"), 1e-6);
        assertEquals(3.0 / Math.sqrt(396.0), (Double) details.get("matthewsCorrelationCoefficient"), 1e-6);
        assertEquals(List.of(
                List.of(2L, 1L, 0L),
                List.of(1L, 1L, 0L),
                List.of(0L, 1L, 0L)),
                details.get("matrix"));
    }

    @Test
    void classificationCohensKappaCorrectsObservedAgreementForChanceAgreement() {
        TrainingMetric metric = TrainingMetrics.classificationKappa().get();

        metric.update(
                GradTensor.of(new float[] {
                        3.0f, 1.0f, 0.0f,
                        2.0f, 0.0f, 1.0f,
                        0.0f, 3.0f, 1.0f,
                        0.0f, 2.0f, 1.0f,
                        3.0f, 2.0f, 0.0f,
                        0.0f, 2.0f, 1.0f
                }, 6, 3),
                GradTensor.of(new float[] {
                        0.0f,
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f,
                        2.0f
                }, 6));

        assertEquals("classification_cohens_kappa", metric.name());
        assertEquals(1.0 / 7.0, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_cohens_kappa", details.get("type"));
        assertEquals(0.5, (Double) details.get("observedAgreement"), 1e-6);
        assertEquals(5.0 / 12.0, (Double) details.get("expectedAgreement"), 1e-6);
        assertEquals(1.0 / 7.0, (Double) details.get("cohensKappa"), 1e-6);
    }

    @Test
    void classificationWeightedMetricsUseSupportWeightedPerClassScores() {
        TrainingMetric precision = TrainingMetrics.classificationWeightedPrecision().get();
        TrainingMetric recall = TrainingMetrics.classificationWeightedRecall().get();
        TrainingMetric f1 = TrainingMetrics.classificationWeightedF1().get();
        GradTensor predictions = GradTensor.of(new float[] {
                3.0f, 1.0f, 0.0f,
                2.0f, 0.0f, 1.0f,
                0.0f, 3.0f, 1.0f,
                0.0f, 2.0f, 1.0f,
                3.0f, 2.0f, 0.0f,
                0.0f, 2.0f, 1.0f
        }, 6, 3);
        GradTensor targets = GradTensor.of(new float[] {
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                1.0f,
                2.0f
        }, 6);

        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);

        assertEquals("classification_weighted_precision", precision.name());
        assertEquals("classification_weighted_recall", recall.name());
        assertEquals("classification_weighted_f1", f1.name());
        assertEquals(4.0 / 9.0, precision.value(), 1e-6);
        assertEquals(0.5, recall.value(), 1e-6);
        assertEquals(7.0 / 15.0, f1.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, f1);
        Map<String, Object> details = detailed.details();
        assertEquals("classification_weighted_f1", details.get("type"));
        assertEquals(4.0 / 9.0, (Double) details.get("weightedPrecision"), 1e-6);
        assertEquals(0.5, (Double) details.get("weightedRecall"), 1e-6);
        assertEquals(7.0 / 15.0, (Double) details.get("weightedF1"), 1e-6);
    }

    @Test
    void classificationBalancedAccuracyAcceptsOneHotTargets() {
        TrainingMetric metric = TrainingMetrics.balancedAccuracy().get();

        metric.update(
                GradTensor.of(new float[] {
                        3.0f, 1.0f, 0.0f,
                        2.0f, 0.0f, 1.0f,
                        0.0f, 3.0f, 1.0f,
                        0.0f, 2.0f, 1.0f,
                        3.0f, 2.0f, 0.0f,
                        0.0f, 2.0f, 1.0f
                }, 6, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f
                }, 6, 3));

        assertEquals(7.0 / 18.0, metric.value(), 1e-6);
    }

    private static void assertDoubleList(List<?> actual, Double first, Double second, Double third) {
        Double[] expected = {first, second, third};
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
