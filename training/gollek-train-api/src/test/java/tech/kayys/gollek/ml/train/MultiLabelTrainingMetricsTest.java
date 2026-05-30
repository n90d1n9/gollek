package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class MultiLabelTrainingMetricsTest {

    @Test
    void multilabelConfusionMatrixReportsPerLabelDetails() {
        TrainingMetric metric = TrainingMetrics.multiLabelConfusionMatrix().get();

        metric.update(
                GradTensor.of(new float[] {
                        2.0f, -1.0f, 0.5f,
                        -2.0f, 3.0f, -0.5f,
                        1.0f, 2.0f, -3.0f
                }, 3, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 1.0f,
                        0.0f, 1.0f, 1.0f,
                        1.0f, 0.0f, 0.0f
                }, 3, 3));

        assertEquals("multilabel_confusion_matrix_macro_f1", metric.name());
        assertEquals(7.0 / 9.0, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("multilabel_confusion_matrix", details.get("type"));
        assertEquals(0.0f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(List.of(0, 1, 2), details.get("labels"));
        assertEquals(3, details.get("labelCount"));
        assertEquals(3L, details.get("samples"));
        assertEquals(9L, details.get("totalLabels"));
        assertEquals(2L, details.get("labelMismatches"));
        assertEquals(1.0 / 3.0, (Double) details.get("exactMatch"), 1e-6);
        assertEquals(2.0 / 9.0, (Double) details.get("hammingLoss"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("macroPrecision"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("macroRecall"), 1e-6);
        assertEquals(7.0 / 9.0, (Double) details.get("macroF1"), 1e-6);
        assertEquals(0.8, (Double) details.get("microPrecision"), 1e-6);
        assertEquals(0.8, (Double) details.get("microRecall"), 1e-6);
        assertEquals(0.8, (Double) details.get("microF1"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("samplePrecision"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("sampleRecall"), 1e-6);
        assertEquals(7.0 / 9.0, (Double) details.get("sampleF1"), 1e-6);
        assertEquals(2.0 / 3.0, (Double) details.get("sampleJaccard"), 1e-6);
        assertEquals("actual_label", details.get("rowMeaning"));
        assertEquals("predicted_label", details.get("columnMeaning"));
        assertEquals(List.of(0, 1), details.get("binaryLabels"));
        assertEquals(List.of(2L, 1L, 1L), details.get("truePositive"));
        assertEquals(List.of(1L, 1L, 1L), details.get("trueNegative"));
        assertEquals(List.of(0L, 1L, 0L), details.get("falsePositive"));
        assertEquals(List.of(0L, 0L, 1L), details.get("falseNegative"));
        assertEquals(List.of(
                List.of(List.of(1L, 0L), List.of(0L, 2L)),
                List.of(List.of(1L, 1L), List.of(0L, 1L)),
                List.of(List.of(1L, 0L), List.of(1L, 1L))),
                details.get("matrixByLabel"));
        assertEquals(List.of(1.0, 0.5, 1.0), details.get("perLabelPrecision"));
        assertEquals(List.of(1.0, 1.0, 0.5), details.get("perLabelRecall"));
        assertEquals(List.of(1.0, 2.0 / 3.0, 2.0 / 3.0), details.get("perLabelF1"));
        assertEquals(List.of(1.0, 0.5, 1.0), details.get("perLabelSpecificity"));
        assertEquals(List.of(1.0, 0.75, 0.75), details.get("perLabelBalancedAccuracy"));
    }

    @Test
    void multilabelConfusionMatrixSupportsCustomThresholdAndShapeGuards() {
        TrainingMetric metric = TrainingMetrics.multiLabelConfusionMatrix(0.5f).get();

        metric.update(
                GradTensor.of(new float[] {
                        0.2f, -0.3f, 1.1f,
                        0.8f, -0.8f, 0.4f
                }, 2, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 1.0f
                }, 2, 3));

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals(0.5f, (Float) details.get("threshold"), 1e-6f);
        assertEquals(2.0 / 9.0, metric.value(), 1e-6);
        assertEquals(List.of(0L, 0L, 1L), details.get("truePositive"));
        assertEquals(List.of(0L, 2L, 0L), details.get("trueNegative"));
        assertEquals(List.of(1L, 0L, 0L), details.get("falsePositive"));
        assertEquals(List.of(1L, 0L, 1L), details.get("falseNegative"));
        assertEquals(0.5, (Double) details.get("microPrecision"), 1e-6);
        assertEquals(1.0 / 3.0, (Double) details.get("microRecall"), 1e-6);
        assertEquals(0.4, (Double) details.get("microF1"), 1e-6);
        assertEquals(0.5, (Double) details.get("samplePrecision"), 1e-6);
        assertEquals(0.25, (Double) details.get("sampleRecall"), 1e-6);
        assertEquals(1.0 / 3.0, (Double) details.get("sampleF1"), 1e-6);
        assertEquals(0.25, (Double) details.get("sampleJaccard"), 1e-6);

        assertThrows(IllegalArgumentException.class,
                () -> TrainingMetrics.multiLabelConfusionMatrix(Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> metric.update(
                        GradTensor.of(new float[] {1.0f, 0.0f}, 2),
                        GradTensor.of(new float[] {1.0f, 0.0f, 1.0f}, 3)));
    }

    @Test
    void multilabelMicroMetricsAggregateAcrossAllLabels() {
        TrainingMetric precision = TrainingMetrics.multiLabelMicroPrecision().get();
        TrainingMetric recall = TrainingMetrics.multiLabelMicroRecall().get();
        TrainingMetric f1 = TrainingMetrics.multiLabelMicroF1().get();
        GradTensor predictions = GradTensor.of(new float[] {
                2.0f, -1.0f, 0.5f,
                -2.0f, 3.0f, -0.5f,
                1.0f, 2.0f, -3.0f
        }, 3, 3);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        }, 3, 3);

        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);

        assertEquals("multilabel_micro_precision", precision.name());
        assertEquals("multilabel_micro_recall", recall.name());
        assertEquals("multilabel_micro_f1", f1.name());
        assertEquals(0.8, precision.value(), 1e-6);
        assertEquals(0.8, recall.value(), 1e-6);
        assertEquals(0.8, f1.value(), 1e-6);
    }

    @Test
    void multilabelMicroMetricsSupportCustomThreshold() {
        TrainingMetric precision = TrainingMetrics.multiLabelMicroPrecision(0.5f).get();
        TrainingMetric recall = TrainingMetrics.multiLabelMicroRecall(0.5f).get();
        TrainingMetric f1 = TrainingMetrics.multiLabelMicroF1(0.5f).get();
        GradTensor predictions = GradTensor.of(new float[] {
                0.2f, -0.3f, 1.1f,
                0.8f, -0.8f, 0.4f
        }, 2, 3);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f,
                0.0f, 0.0f, 1.0f
        }, 2, 3);

        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);

        assertEquals(0.5, precision.value(), 1e-6);
        assertEquals(1.0 / 3.0, recall.value(), 1e-6);
        assertEquals(0.4, f1.value(), 1e-6);
    }

    @Test
    void multilabelSampleMetricsAveragePerExampleScores() {
        TrainingMetric precision = TrainingMetrics.multiLabelSamplePrecision().get();
        TrainingMetric recall = TrainingMetrics.multiLabelSampleRecall().get();
        TrainingMetric f1 = TrainingMetrics.multiLabelSampleF1().get();
        TrainingMetric jaccard = TrainingMetrics.multiLabelSampleJaccard().get();
        GradTensor predictions = GradTensor.of(new float[] {
                2.0f, -1.0f, 0.5f,
                -2.0f, 3.0f, -0.5f,
                1.0f, 2.0f, -3.0f
        }, 3, 3);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        }, 3, 3);

        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);
        jaccard.update(predictions, targets);

        assertEquals("multilabel_sample_precision", precision.name());
        assertEquals("multilabel_sample_recall", recall.name());
        assertEquals("multilabel_sample_f1", f1.name());
        assertEquals("multilabel_sample_jaccard", jaccard.name());
        assertEquals(5.0 / 6.0, precision.value(), 1e-6);
        assertEquals(5.0 / 6.0, recall.value(), 1e-6);
        assertEquals(7.0 / 9.0, f1.value(), 1e-6);
        assertEquals(2.0 / 3.0, jaccard.value(), 1e-6);
    }

    @Test
    void multilabelSampleMetricsSupportCustomThreshold() {
        TrainingMetric precision = TrainingMetrics.multiLabelSamplePrecision(0.5f).get();
        TrainingMetric recall = TrainingMetrics.multiLabelSampleRecall(0.5f).get();
        TrainingMetric f1 = TrainingMetrics.multiLabelSampleF1(0.5f).get();
        TrainingMetric jaccard = TrainingMetrics.multiLabelSampleJaccard(0.5f).get();
        GradTensor predictions = GradTensor.of(new float[] {
                0.2f, -0.3f, 1.1f,
                0.8f, -0.8f, 0.4f
        }, 2, 3);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f,
                0.0f, 0.0f, 1.0f
        }, 2, 3);

        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);
        jaccard.update(predictions, targets);

        assertEquals(0.5, precision.value(), 1e-6);
        assertEquals(0.25, recall.value(), 1e-6);
        assertEquals(1.0 / 3.0, f1.value(), 1e-6);
        assertEquals(0.25, jaccard.value(), 1e-6);
    }

    @Test
    void multilabelSampleRankingMetricsScoreEachExampleWithoutThresholding() {
        TrainingMetric lrap = TrainingMetrics.multiLabelLabelRankingAveragePrecision().get();
        TrainingMetric lrapAlias = TrainingMetrics.multiLabelLrap().get();
        TrainingMetric rankingLoss = TrainingMetrics.multiLabelRankingLoss().get();
        TrainingMetric coverageError = TrainingMetrics.multiLabelCoverageError().get();
        GradTensor predictions = GradTensor.of(new float[] {
                0.9f, 0.2f, 0.7f, 0.1f,
                0.1f, 0.8f, 0.4f, 0.7f,
                0.5f, 0.5f, 0.4f, 0.3f
        }, 3, 4);
        GradTensor targets = GradTensor.of(new float[] {
                1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 0.0f, 0.0f
        }, 3, 4);

        lrap.update(predictions, targets);
        lrapAlias.update(predictions, targets);
        rankingLoss.update(predictions, targets);
        coverageError.update(predictions, targets);

        assertEquals("multilabel_label_ranking_average_precision", lrap.name());
        assertEquals("multilabel_label_ranking_average_precision", lrapAlias.name());
        assertEquals("multilabel_ranking_loss", rankingLoss.name());
        assertEquals("multilabel_coverage_error", coverageError.name());
        assertEquals(23.0 / 36.0, lrap.value(), 1e-6);
        assertEquals(lrap.value(), lrapAlias.value(), 1e-6);
        assertEquals(7.0 / 18.0, rankingLoss.value(), 1e-6);
        assertEquals(8.0 / 3.0, coverageError.value(), 1e-6);
    }

    @Test
    void multilabelRankingMetricsHandleRowsWithoutComparablePairs() {
        TrainingMetric lrap = TrainingMetrics.multiLabelLrap().get();
        TrainingMetric rankingLoss = TrainingMetrics.multiLabelRankingLoss().get();
        TrainingMetric coverageError = TrainingMetrics.multiLabelCoverageError().get();
        GradTensor predictions = GradTensor.of(new float[] {
                0.2f, 0.1f, 0.0f,
                0.4f, 0.3f, 0.2f
        }, 2, 3);
        GradTensor targets = GradTensor.of(new float[] {
                0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f
        }, 2, 3);

        lrap.update(predictions, targets);
        rankingLoss.update(predictions, targets);
        coverageError.update(predictions, targets);

        assertEquals(1.0, lrap.value(), 1e-6);
        assertEquals(0.0, rankingLoss.value(), 1e-6);
        assertEquals(1.5, coverageError.value(), 1e-6);
    }

    @Test
    void multilabelBestF1ThresholdsTuneEachLabelIndependently() {
        TrainingMetric metric = TrainingMetrics.multiLabelBestF1Thresholds().get();

        metric.update(
                GradTensor.of(new float[] {
                        0.9f, 0.1f, 0.2f,
                        0.8f, 0.7f, 0.6f,
                        0.2f, 0.6f, 0.4f
                }, 3, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 1.0f,
                        0.0f, 1.0f, 0.0f
                }, 3, 3));

        assertEquals("multilabel_macro_best_f1", metric.name());
        assertEquals(8.0 / 9.0, metric.value(), 1e-6);

        DetailedTrainingMetric detailed = assertInstanceOf(DetailedTrainingMetric.class, metric);
        Map<String, Object> details = detailed.details();
        assertEquals("multilabel_threshold_optimization", details.get("type"));
        assertEquals("f1", details.get("objective"));
        assertEquals("macro", details.get("aggregation"));
        assertEquals(Boolean.TRUE, details.get("defined"));
        assertEquals(3L, details.get("samples"));
        assertEquals(3, details.get("labelCount"));
        assertEquals(List.of(0, 1, 2), details.get("labels"));
        assertEquals(8.0 / 9.0, (Double) details.get("macroF1"), 1e-6);
        assertEquals(5.0 / 6.0, (Double) details.get("macroPrecision"), 1e-6);
        assertEquals(1.0, (Double) details.get("macroRecall"), 1e-6);
        assertEquals(List.of(0.8f, 0.6f, 0.6f), details.get("perLabelThreshold"));
        assertEquals(List.of(1.0, 2.0 / 3.0, 1.0), details.get("perLabelF1"));
        assertEquals(List.of(1.0, 0.5, 1.0), details.get("perLabelPrecision"));
        assertEquals(List.of(1.0, 1.0, 1.0), details.get("perLabelRecall"));
        assertEquals(List.of(2L, 1L, 1L), details.get("truePositive"));
        assertEquals(List.of(1L, 1L, 2L), details.get("trueNegative"));
        assertEquals(List.of(0L, 1L, 0L), details.get("falsePositive"));
        assertEquals(List.of(0L, 0L, 0L), details.get("falseNegative"));
        assertEquals(List.of(
                List.of(List.of(1L, 0L), List.of(0L, 2L)),
                List.of(List.of(1L, 1L), List.of(0L, 1L)),
                List.of(List.of(2L, 0L), List.of(0L, 1L))),
                details.get("matrixByLabel"));
    }
}
