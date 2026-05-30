package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class RegressionTrainingMetricsTest {

    @Test
    void extendedRegressionMetricsStreamAcrossBatches() {
        TrainingMetric mape = TrainingMetrics.mape().get();
        TrainingMetric smape = TrainingMetrics.smape().get();
        TrainingMetric meanBiasError = TrainingMetrics.mbe().get();
        TrainingMetric explainedVariance = TrainingMetrics.explainedVariance().get();
        TrainingMetric medianAbsoluteError = TrainingMetrics.medae().get();
        TrainingMetric maxError = TrainingMetrics.maxError().get();

        GradTensor firstPredictions = GradTensor.of(new float[] {0.0f, 2.0f}, 2);
        GradTensor firstTargets = GradTensor.of(new float[] {0.0f, 1.0f}, 2);
        GradTensor secondPredictions = GradTensor.of(new float[] {4.0f, 6.0f}, 2);
        GradTensor secondTargets = GradTensor.of(new float[] {5.0f, 10.0f}, 2);

        mape.update(firstPredictions, firstTargets);
        mape.update(secondPredictions, secondTargets);
        smape.update(firstPredictions, firstTargets);
        smape.update(secondPredictions, secondTargets);
        meanBiasError.update(firstPredictions, firstTargets);
        meanBiasError.update(secondPredictions, secondTargets);
        explainedVariance.update(firstPredictions, firstTargets);
        explainedVariance.update(secondPredictions, secondTargets);
        medianAbsoluteError.update(firstPredictions, firstTargets);
        medianAbsoluteError.update(secondPredictions, secondTargets);
        maxError.update(firstPredictions, firstTargets);
        maxError.update(secondPredictions, secondTargets);

        assertEquals("mape", mape.name());
        assertEquals(8.0 / 15.0, mape.value(), 1e-6);
        assertEquals("smape", smape.name());
        assertEquals(25.0 / 72.0, smape.value(), 1e-6);
        assertEquals("mbe", meanBiasError.name());
        assertEquals(-1.0, meanBiasError.value(), 1e-6);
        assertEquals("explained_variance", explainedVariance.name());
        assertEquals(24.0 / 31.0, explainedVariance.value(), 1e-6);
        assertEquals("median_absolute_error", medianAbsoluteError.name());
        assertEquals(1.0, medianAbsoluteError.value(), 1e-6);
        assertEquals("max_error", maxError.name());
        assertEquals(4.0, maxError.value(), 1e-6);

        DetailedTrainingMetric detailedMape = assertInstanceOf(DetailedTrainingMetric.class, mape);
        Map<String, Object> mapeDetails = detailedMape.details();
        assertEquals("fraction", mapeDetails.get("scale"));
        assertEquals(3L, mapeDetails.get("samples"));
        assertEquals(1L, mapeDetails.get("skippedZeroTargets"));

        DetailedTrainingMetric detailedSmape = assertInstanceOf(DetailedTrainingMetric.class, smape);
        assertEquals(4L, detailedSmape.details().get("samples"));
        assertEquals(1L, detailedSmape.details().get("zeroDenominatorCount"));

        DetailedTrainingMetric detailedExplainedVariance =
                assertInstanceOf(DetailedTrainingMetric.class, explainedVariance);
        assertEquals(15.5, (Double) detailedExplainedVariance.details().get("targetVariance"), 1e-6);
        assertEquals(3.5, (Double) detailedExplainedVariance.details().get("residualVariance"), 1e-6);

        DetailedTrainingMetric detailedMedianAbsoluteError =
                assertInstanceOf(DetailedTrainingMetric.class, medianAbsoluteError);
        assertEquals(Boolean.TRUE, detailedMedianAbsoluteError.details().get("exact"));

        DetailedTrainingMetric detailedMaxError = assertInstanceOf(DetailedTrainingMetric.class, maxError);
        assertEquals(-4.0, (Double) detailedMaxError.details().get("signedErrorAtMax"), 1e-6);
    }

    @Test
    void logScaleRegressionMetricsStreamAndRejectNegativeInputs() {
        TrainingMetric msle = TrainingMetrics.msle().get();
        TrainingMetric rmsle = TrainingMetrics.rmsle().get();
        GradTensor firstPredictions = GradTensor.of(new float[] {0.0f, 2.0f}, 2);
        GradTensor firstTargets = GradTensor.of(new float[] {0.0f, 1.0f}, 2);
        GradTensor secondPredictions = GradTensor.of(new float[] {4.0f, 6.0f}, 2);
        GradTensor secondTargets = GradTensor.of(new float[] {5.0f, 10.0f}, 2);

        msle.update(firstPredictions, firstTargets);
        msle.update(secondPredictions, secondTargets);
        rmsle.update(firstPredictions, firstTargets);
        rmsle.update(secondPredictions, secondTargets);

        double expectedMsle = (
                square(Math.log1p(0.0) - Math.log1p(0.0))
                        + square(Math.log1p(2.0) - Math.log1p(1.0))
                        + square(Math.log1p(4.0) - Math.log1p(5.0))
                        + square(Math.log1p(6.0) - Math.log1p(10.0))) / 4.0;
        assertEquals("msle", msle.name());
        assertEquals(expectedMsle, msle.value(), 1e-6);
        assertEquals("rmsle", rmsle.name());
        assertEquals(Math.sqrt(expectedMsle), rmsle.value(), 1e-6);

        DetailedTrainingMetric detailedMsle = assertInstanceOf(DetailedTrainingMetric.class, msle);
        assertEquals("log1p", detailedMsle.details().get("logTransform"));
        assertEquals("predictions >= 0 and targets >= 0", detailedMsle.details().get("inputDomain"));

        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.msle().get().update(
                GradTensor.of(new float[] {-0.01f}, 1),
                GradTensor.of(new float[] {1.0f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.rmsle().get().update(
                GradTensor.of(new float[] {1.0f}, 1),
                GradTensor.of(new float[] {-0.01f}, 1)));
    }

    @Test
    void pinballLossSupportsAsymmetricQuantileRegressionEvaluation() {
        TrainingMetric median = TrainingMetrics.pinballLoss(0.5).get();
        TrainingMetric upper = TrainingMetrics.meanPinballLoss(0.9).get();
        GradTensor firstPredictions = GradTensor.of(new float[] {0.0f, 2.0f}, 2);
        GradTensor firstTargets = GradTensor.of(new float[] {0.0f, 1.0f}, 2);
        GradTensor secondPredictions = GradTensor.of(new float[] {4.0f, 6.0f}, 2);
        GradTensor secondTargets = GradTensor.of(new float[] {5.0f, 10.0f}, 2);

        median.update(firstPredictions, firstTargets);
        median.update(secondPredictions, secondTargets);
        upper.update(firstPredictions, firstTargets);
        upper.update(secondPredictions, secondTargets);

        assertEquals("pinball_loss_q50", median.name());
        assertEquals(0.75, median.value(), 1e-6);
        assertEquals("pinball_loss_q90", upper.name());
        assertEquals(1.15, upper.value(), 1e-6);

        DetailedTrainingMetric detailedUpper = assertInstanceOf(DetailedTrainingMetric.class, upper);
        Map<String, Object> upperDetails = detailedUpper.details();
        assertEquals("pinball_loss", upperDetails.get("type"));
        assertEquals("pinball_loss_q90", upperDetails.get("metricName"));
        assertEquals(0.9, (Double) upperDetails.get("quantile"), 1e-6);
        assertEquals(4L, upperDetails.get("samples"));
    }

    @Test
    void pinballLossRejectsInvalidQuantiles() {
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.pinballLoss(0.0));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.pinballLoss(1.0));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.pinballLoss(Double.NaN));
    }

    @Test
    void predictionIntervalMetricsReportCoverageWidthAndCrossing() {
        TrainingMetric coverage = TrainingMetrics.picp().get();
        TrainingMetric meanWidth = TrainingMetrics.predictionIntervalMeanWidth().get();
        TrainingMetric normalizedMeanWidth = TrainingMetrics.predictionIntervalNormalizedMeanWidth().get();
        GradTensor intervals = GradTensor.of(new float[] {
                -1.0f, 1.0f,
                1.5f, 2.5f,
                4.0f, 6.0f,
                12.0f, 8.0f
        }, 4, 2);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 1.0f, 5.0f, 10.0f}, 4);

        coverage.update(intervals, targets);
        meanWidth.update(intervals, targets);
        normalizedMeanWidth.update(intervals, targets);

        assertEquals("prediction_interval_coverage", coverage.name());
        assertEquals(0.75, coverage.value(), 1e-6);
        assertEquals("prediction_interval_mean_width", meanWidth.name());
        assertEquals(2.25, meanWidth.value(), 1e-6);
        assertEquals("prediction_interval_normalized_mean_width", normalizedMeanWidth.name());
        assertEquals(0.225, normalizedMeanWidth.value(), 1e-6);

        DetailedTrainingMetric detailedCoverage = assertInstanceOf(DetailedTrainingMetric.class, coverage);
        Map<String, Object> details = detailedCoverage.details();
        assertEquals(4L, details.get("samples"));
        assertEquals(3L, details.get("coveredIntervals"));
        assertEquals(1L, details.get("crossedIntervals"));
        assertEquals(0.25, (Double) details.get("crossedFraction"), 1e-6);
        assertEquals(Boolean.TRUE, details.get("boundsReorderedForEvaluation"));
        assertEquals(10.0, (Double) details.get("targetRange"), 1e-6);
    }

    @Test
    void predictionIntervalMetricsRequireLowerUpperPredictionPairs() {
        TrainingMetric coverage = TrainingMetrics.predictionIntervalCoverage().get();
        GradTensor scalarPredictions = GradTensor.of(new float[] {1.0f, 2.0f}, 2);
        GradTensor targets = GradTensor.of(new float[] {1.5f, 2.5f}, 2);

        assertThrows(IllegalArgumentException.class, () -> coverage.update(scalarPredictions, targets));
    }

    private static double square(double value) {
        return value * value;
    }
}
