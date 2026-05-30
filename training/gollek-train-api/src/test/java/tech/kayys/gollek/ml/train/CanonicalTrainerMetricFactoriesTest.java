package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.util.Map;
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
    void nestedMetricsFacadeExposesExtendedRegressionMetrics() {
        CanonicalTrainer.Metric mape = CanonicalTrainer.Metrics.meanAbsolutePercentageError().get();
        CanonicalTrainer.Metric smape = CanonicalTrainer.Metrics.smape().get();
        CanonicalTrainer.Metric bias = CanonicalTrainer.Metrics.meanBiasError().get();
        CanonicalTrainer.Metric explainedVariance = CanonicalTrainer.Metrics.explainedVarianceScore().get();
        CanonicalTrainer.Metric medianAbsoluteError = CanonicalTrainer.Metrics.medae().get();
        CanonicalTrainer.Metric maxError = CanonicalTrainer.Metrics.maxError().get();
        CanonicalTrainer.Metric pinballQ90 = CanonicalTrainer.Metrics.meanPinballLoss(0.9).get();
        CanonicalTrainer.Metric intervalCoverage = CanonicalTrainer.Metrics.picp().get();
        CanonicalTrainer.Metric intervalMeanWidth = CanonicalTrainer.Metrics.predictionIntervalMeanWidth().get();
        GradTensor predictions = GradTensor.of(new float[] {0.0f, 2.0f, 4.0f, 6.0f}, 4);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 1.0f, 5.0f, 10.0f}, 4);
        GradTensor intervalPredictions = GradTensor.of(new float[] {
                -1.0f, 1.0f,
                1.5f, 2.5f,
                4.0f, 6.0f,
                12.0f, 8.0f
        }, 4, 2);

        mape.update(predictions, targets);
        smape.update(predictions, targets);
        bias.update(predictions, targets);
        explainedVariance.update(predictions, targets);
        medianAbsoluteError.update(predictions, targets);
        maxError.update(predictions, targets);
        pinballQ90.update(predictions, targets);
        intervalCoverage.update(intervalPredictions, targets);
        intervalMeanWidth.update(intervalPredictions, targets);

        assertEquals("mape", mape.name());
        assertEquals(8.0 / 15.0, mape.value(), 1e-6);
        assertEquals("smape", smape.name());
        assertEquals(25.0 / 72.0, smape.value(), 1e-6);
        assertEquals("mbe", bias.name());
        assertEquals(-1.0, bias.value(), 1e-6);
        assertEquals("explained_variance", explainedVariance.name());
        assertEquals(24.0 / 31.0, explainedVariance.value(), 1e-6);
        assertEquals("median_absolute_error", medianAbsoluteError.name());
        assertEquals(1.0, medianAbsoluteError.value(), 1e-6);
        assertEquals("max_error", maxError.name());
        assertEquals(4.0, maxError.value(), 1e-6);
        assertEquals("pinball_loss_q90", pinballQ90.name());
        assertEquals(1.15, pinballQ90.value(), 1e-6);
        assertEquals("prediction_interval_coverage", intervalCoverage.name());
        assertEquals(0.75, intervalCoverage.value(), 1e-6);
        assertEquals("prediction_interval_mean_width", intervalMeanWidth.name());
        assertEquals(2.25, intervalMeanWidth.value(), 1e-6);

        CanonicalTrainer.DetailedMetric detailedMape =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, mape);
        assertEquals(1L, detailedMape.details().get("skippedZeroTargets"));
        CanonicalTrainer.DetailedMetric detailedMaxError =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, maxError);
        assertEquals(-4.0, (Double) detailedMaxError.details().get("signedErrorAtMax"), 1e-6);
        CanonicalTrainer.DetailedMetric detailedPinball =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, pinballQ90);
        assertEquals(0.9, (Double) detailedPinball.details().get("quantile"), 1e-6);
        CanonicalTrainer.DetailedMetric detailedInterval =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, intervalCoverage);
        assertEquals(1L, detailedInterval.details().get("crossedIntervals"));
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

    @Test
    void nestedMetricsFacadeExposesImbalanceMetrics() {
        CanonicalTrainer.Metric classification =
                CanonicalTrainer.Metrics.classificationBalancedAccuracy().get();
        CanonicalTrainer.Metric classificationMcc =
                CanonicalTrainer.Metrics.classificationMcc().get();
        CanonicalTrainer.Metric classificationWeightedF1 =
                CanonicalTrainer.Metrics.classificationWeightedF1().get();
        CanonicalTrainer.Metric classificationKappa =
                CanonicalTrainer.Metrics.classificationKappa().get();
        GradTensor classificationPredictions = GradTensor.of(new float[] {
                3.0f, 1.0f, 0.0f,
                2.0f, 0.0f, 1.0f,
                0.0f, 3.0f, 1.0f,
                0.0f, 2.0f, 1.0f,
                3.0f, 2.0f, 0.0f,
                0.0f, 2.0f, 1.0f
        }, 6, 3);
        GradTensor classificationTargets =
                GradTensor.of(new float[] {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f}, 6);
        classification.update(
                classificationPredictions,
                classificationTargets);
        classificationMcc.update(
                classificationPredictions,
                classificationTargets);
        classificationWeightedF1.update(
                classificationPredictions,
                classificationTargets);
        classificationKappa.update(
                classificationPredictions,
                classificationTargets);

        CanonicalTrainer.Metric binary = CanonicalTrainer.Metrics.binaryBalancedAccuracy(1.0f).get();
        CanonicalTrainer.Metric binaryMcc = CanonicalTrainer.Metrics.binaryMcc(1.0f).get();
        CanonicalTrainer.Metric binaryKappa = CanonicalTrainer.Metrics.binaryKappa(1.0f).get();
        GradTensor binaryPredictions = GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4);
        GradTensor binaryTargets = GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4);
        binary.update(
                binaryPredictions,
                binaryTargets);
        binaryMcc.update(
                binaryPredictions,
                binaryTargets);
        binaryKappa.update(
                binaryPredictions,
                binaryTargets);

        CanonicalTrainer.DetailedMetric classificationDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, classification);
        CanonicalTrainer.DetailedMetric classificationMccDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, classificationMcc);
        CanonicalTrainer.DetailedMetric classificationWeightedF1Details =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, classificationWeightedF1);
        CanonicalTrainer.DetailedMetric classificationKappaDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, classificationKappa);
        CanonicalTrainer.DetailedMetric binaryDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, binary);
        CanonicalTrainer.DetailedMetric binaryMccDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, binaryMcc);
        CanonicalTrainer.DetailedMetric binaryKappaDetails =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, binaryKappa);
        assertEquals("classification_balanced_accuracy", classification.name());
        assertEquals(7.0 / 18.0, classification.value(), 1e-6);
        assertEquals("classification_balanced_accuracy", classificationDetails.details().get("type"));
        assertEquals(java.util.List.of(3L, 2L, 1L), classificationDetails.details().get("support"));
        assertEquals("classification_matthews_correlation_coefficient", classificationMcc.name());
        assertEquals(3.0 / Math.sqrt(396.0), classificationMcc.value(), 1e-6);
        assertEquals(
                "classification_matthews_correlation_coefficient",
                classificationMccDetails.details().get("type"));
        assertEquals("classification_weighted_f1", classificationWeightedF1.name());
        assertEquals(7.0 / 15.0, classificationWeightedF1.value(), 1e-6);
        assertEquals("classification_weighted_f1", classificationWeightedF1Details.details().get("type"));
        assertEquals(4.0 / 9.0, (Double) classificationWeightedF1Details.details().get("weightedPrecision"), 1e-6);
        assertEquals("classification_cohens_kappa", classificationKappa.name());
        assertEquals(1.0 / 7.0, classificationKappa.value(), 1e-6);
        assertEquals("classification_cohens_kappa", classificationKappaDetails.details().get("type"));
        assertEquals("binary_balanced_accuracy", binary.name());
        assertEquals(0.75, binary.value(), 1e-6);
        assertEquals("binary_balanced_accuracy", binaryDetails.details().get("type"));
        assertEquals(1.0f, (Float) binaryDetails.details().get("threshold"), 1e-6f);
        assertEquals("binary_matthews_correlation_coefficient", binaryMcc.name());
        assertEquals(1.0 / Math.sqrt(3.0), binaryMcc.value(), 1e-6);
        assertEquals("binary_matthews_correlation_coefficient", binaryMccDetails.details().get("type"));
        assertEquals("binary_cohens_kappa", binaryKappa.name());
        assertEquals(0.5, binaryKappa.value(), 1e-6);
        assertEquals("binary_cohens_kappa", binaryKappaDetails.details().get("type"));
    }

    @Test
    void nestedMetricsFacadeExposesBinaryBestF1Threshold() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.binaryBestF1Threshold().get();

        metric.update(
                GradTensor.of(new float[] {2.0f, 0.5f, -1.0f, -2.0f}, 4),
                GradTensor.of(new float[] {1.0f, 0.0f, 0.0f, 1.0f}, 4));

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals("binary_best_f1", metric.name());
        assertEquals(2.0 / 3.0, metric.value(), 1e-6);
        assertEquals("binary_threshold_optimization", detailedMetric.details().get("type"));
        assertEquals(2.0f, (Float) detailedMetric.details().get("threshold"), 1e-6f);
    }

    @Test
    void nestedMetricsFacadeExposesBinaryCalibrationMetrics() {
        CanonicalTrainer.Metric brier = CanonicalTrainer.Metrics.binaryBrierScore().get();
        CanonicalTrainer.Metric ece = CanonicalTrainer.Metrics.binaryExpectedCalibrationError(5).get();
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

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, ece);
        assertEquals("binary_brier_score", brier.name());
        assertEquals("binary_expected_calibration_error", ece.name());
        assertEquals(0.045625, brier.value(), 1e-6);
        assertEquals(0.1875, ece.value(), 1e-6);
        assertEquals("binary_calibration", detailedMetric.details().get("type"));
        assertEquals(5, detailedMetric.details().get("bins"));
    }

    @Test
    void nestedMetricsFacadeExposesClassificationCalibrationMetrics() {
        CanonicalTrainer.Metric brier = CanonicalTrainer.Metrics.classificationBrierScore().get();
        CanonicalTrainer.Metric ece =
                CanonicalTrainer.Metrics.classificationExpectedCalibrationError(5).get();
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

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, ece);
        Map<String, Object> details = detailedMetric.details();
        assertEquals("classification_brier_score", brier.name());
        assertEquals("classification_expected_calibration_error", ece.name());
        assertEquals(1.06 / 3.0, brier.value(), 1e-6);
        assertEquals(1.0 / 15.0, ece.value(), 1e-6);
        assertEquals("classification_calibration", details.get("type"));
        assertEquals("top_label", details.get("mode"));
        assertEquals(5, details.get("bins"));
        assertEquals(java.util.List.of(0L, 0L, 2L, 0L, 1L), details.get("binCount"));
    }

    @Test
    void nestedMetricsFacadeExposesMultilabelConfusionMatrix() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.multiLabelConfusionMatrix(0.5f).get();

        metric.update(
                GradTensor.of(new float[] {
                        0.2f, -0.3f, 1.1f,
                        0.8f, -0.8f, 0.4f
                }, 2, 3),
                GradTensor.of(new float[] {
                        1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 1.0f
                }, 2, 3));

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals("multilabel_confusion_matrix_macro_f1", metric.name());
        assertEquals(2.0 / 9.0, metric.value(), 1e-6);
        assertEquals("multilabel_confusion_matrix", detailedMetric.details().get("type"));
        assertEquals(0.5f, (Float) detailedMetric.details().get("threshold"), 1e-6f);
        assertEquals(java.util.List.of(0L, 0L, 1L), detailedMetric.details().get("truePositive"));
        assertEquals(java.util.List.of(0L, 2L, 0L), detailedMetric.details().get("trueNegative"));
    }

    @Test
    void nestedMetricsFacadeExposesMultilabelMicroMetrics() {
        CanonicalTrainer.Metric precision = CanonicalTrainer.Metrics.multiLabelMicroPrecision(0.5f).get();
        CanonicalTrainer.Metric recall = CanonicalTrainer.Metrics.multiLabelMicroRecall(0.5f).get();
        CanonicalTrainer.Metric f1 = CanonicalTrainer.Metrics.multiLabelMicroF1(0.5f).get();
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

        assertEquals("multilabel_micro_precision", precision.name());
        assertEquals("multilabel_micro_recall", recall.name());
        assertEquals("multilabel_micro_f1", f1.name());
        assertEquals(0.5, precision.value(), 1e-6);
        assertEquals(1.0 / 3.0, recall.value(), 1e-6);
        assertEquals(0.4, f1.value(), 1e-6);
    }

    @Test
    void nestedMetricsFacadeExposesMultilabelSampleMetrics() {
        CanonicalTrainer.Metric precision = CanonicalTrainer.Metrics.multiLabelSamplePrecision(0.5f).get();
        CanonicalTrainer.Metric recall = CanonicalTrainer.Metrics.multiLabelSampleRecall(0.5f).get();
        CanonicalTrainer.Metric f1 = CanonicalTrainer.Metrics.multiLabelSampleF1(0.5f).get();
        CanonicalTrainer.Metric jaccard = CanonicalTrainer.Metrics.multiLabelSampleJaccard(0.5f).get();
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

        assertEquals("multilabel_sample_precision", precision.name());
        assertEquals("multilabel_sample_recall", recall.name());
        assertEquals("multilabel_sample_f1", f1.name());
        assertEquals("multilabel_sample_jaccard", jaccard.name());
        assertEquals(0.5, precision.value(), 1e-6);
        assertEquals(0.25, recall.value(), 1e-6);
        assertEquals(1.0 / 3.0, f1.value(), 1e-6);
        assertEquals(0.25, jaccard.value(), 1e-6);
    }

    @Test
    void nestedMetricsFacadeExposesMultilabelRankingMetrics() {
        CanonicalTrainer.Metric lrap = CanonicalTrainer.Metrics.multiLabelLrap().get();
        CanonicalTrainer.Metric rankingLoss = CanonicalTrainer.Metrics.multiLabelRankingLoss().get();
        CanonicalTrainer.Metric coverageError = CanonicalTrainer.Metrics.multiLabelCoverageError().get();
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
        rankingLoss.update(predictions, targets);
        coverageError.update(predictions, targets);

        assertEquals("multilabel_label_ranking_average_precision", lrap.name());
        assertEquals("multilabel_ranking_loss", rankingLoss.name());
        assertEquals("multilabel_coverage_error", coverageError.name());
        assertEquals(23.0 / 36.0, lrap.value(), 1e-6);
        assertEquals(7.0 / 18.0, rankingLoss.value(), 1e-6);
        assertEquals(8.0 / 3.0, coverageError.value(), 1e-6);
    }

    @Test
    void nestedMetricsFacadeExposesMultilabelThresholdTuning() {
        CanonicalTrainer.Metric metric = CanonicalTrainer.Metrics.multiLabelBestF1Thresholds().get();

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

        CanonicalTrainer.DetailedMetric detailedMetric =
                assertInstanceOf(CanonicalTrainer.DetailedMetric.class, metric);
        assertEquals("multilabel_macro_best_f1", metric.name());
        assertEquals(8.0 / 9.0, metric.value(), 1e-6);
        assertEquals("multilabel_threshold_optimization", detailedMetric.details().get("type"));
        assertEquals(java.util.List.of(0.8f, 0.6f, 0.6f), detailedMetric.details().get("perLabelThreshold"));
    }

    private static float logit(double probability) {
        return (float) Math.log(probability / (1.0 - probability));
    }

    private static float logProb(double probability) {
        return (float) Math.log(probability);
    }
}
