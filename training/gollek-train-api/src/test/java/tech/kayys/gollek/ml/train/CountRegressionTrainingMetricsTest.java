package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;

class CountRegressionTrainingMetricsTest {

    @Test
    void poissonDevianceSupportsRawMeansAndLogRates() {
        TrainingMetric raw = TrainingMetrics.meanPoissonDeviance().get();
        TrainingMetric logRate = TrainingMetrics.poissonLogRateDeviance().get();
        GradTensor rawMeans = GradTensor.of(new float[] {0.5f, 1.0f, 2.0f}, 3);
        GradTensor logRates = GradTensor.of(new float[] {
                (float) Math.log(0.5),
                (float) Math.log(1.0),
                (float) Math.log(2.0)
        }, 3);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 1.0f, 3.0f}, 3);

        raw.update(rawMeans, targets);
        logRate.update(logRates, targets);

        double expected = (poissonDeviance(0.0, 0.5)
                + poissonDeviance(1.0, 1.0)
                + poissonDeviance(3.0, 2.0)) / 3.0;
        assertEquals("mean_poisson_deviance", raw.name());
        assertEquals(expected, raw.value(), 1e-6);
        assertEquals(expected, logRate.value(), 1e-6);

        DetailedTrainingMetric details = assertInstanceOf(DetailedTrainingMetric.class, logRate);
        Map<String, Object> detailMap = details.details();
        assertEquals(Boolean.TRUE, detailMap.get("logInput"));
        assertEquals(3L, detailMap.get("samples"));
        assertEquals(4.0 / 3.0, (Double) detailMap.get("meanTarget"), 1e-6);
        assertEquals(3.5 / 3.0, (Double) detailMap.get("meanPrediction"), 1e-6);
    }

    @Test
    void poissonDevianceIsAvailableThroughGollekFacadeAndRejectsInvalidDomain() {
        TrainingMetric metric = Gollek.DL.meanPoissonDevianceMetric(false, 1e-6).get();
        metric.update(
                GradTensor.of(new float[] {0.0f}, 1),
                GradTensor.of(new float[] {1.0f}, 1));
        assertEquals(2.0 * (Math.log(1.0 / 1e-6) - 1.0 + 1e-6), metric.value(), 1e-6);

        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.poissonDeviance().get().update(
                GradTensor.of(new float[] {-0.1f}, 1),
                GradTensor.of(new float[] {1.0f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.poissonDeviance().get().update(
                GradTensor.of(new float[] {1.0f}, 1),
                GradTensor.of(new float[] {-0.1f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanPoissonDeviance(false, 0.0));
    }

    @Test
    void tweedieDevianceSupportsCompoundPoissonGammaAndBoundaryPowers() {
        TrainingMetric compound = TrainingMetrics.compoundPoissonGammaDeviance().get();
        TrainingMetric raw = TrainingMetrics.meanTweedieDeviance(1.5, false).get();
        TrainingMetric poisson = TrainingMetrics.meanTweedieDeviance(1.0).get();
        TrainingMetric gamma = TrainingMetrics.meanTweedieDeviance(2.0).get();
        GradTensor means = GradTensor.of(new float[] {0.5f, 1.0f, 2.0f}, 3);
        GradTensor logMeans = GradTensor.of(new float[] {
                (float) Math.log(0.5),
                (float) Math.log(1.0),
                (float) Math.log(2.0)
        }, 3);
        GradTensor targets = GradTensor.of(new float[] {0.0f, 1.0f, 3.0f}, 3);
        GradTensor positiveTargets = GradTensor.of(new float[] {0.5f, 1.0f, 3.0f}, 3);

        compound.update(logMeans, targets);
        raw.update(means, targets);
        poisson.update(logMeans, targets);
        gamma.update(logMeans, positiveTargets);

        double expectedCompound = (tweedieDeviance(0.0, 0.5, 1.5)
                + tweedieDeviance(1.0, 1.0, 1.5)
                + tweedieDeviance(3.0, 2.0, 1.5)) / 3.0;
        double expectedPoisson = (poissonDeviance(0.0, 0.5)
                + poissonDeviance(1.0, 1.0)
                + poissonDeviance(3.0, 2.0)) / 3.0;
        double expectedGamma = (gammaDeviance(0.5, 0.5)
                + gammaDeviance(1.0, 1.0)
                + gammaDeviance(3.0, 2.0)) / 3.0;

        assertEquals("mean_tweedie_deviance", compound.name());
        assertEquals(expectedCompound, compound.value(), 1e-6);
        assertEquals(expectedCompound, raw.value(), 1e-6);
        assertEquals(expectedPoisson, poisson.value(), 1e-6);
        assertEquals(expectedGamma, gamma.value(), 1e-6);

        DetailedTrainingMetric details = assertInstanceOf(DetailedTrainingMetric.class, compound);
        Map<String, Object> detailMap = details.details();
        assertEquals(1.5, (Double) detailMap.get("power"), 1e-6);
        assertEquals(Boolean.TRUE, detailMap.get("logInput"));
        assertEquals("mean_tweedie_deviance", detailMap.get("type"));
    }

    @Test
    void tweedieDevianceIsAvailableThroughGollekFacadeAndRejectsInvalidDomain() {
        TrainingMetric metric = Gollek.DL.meanTweedieDevianceMetric(1.5, false, 1e-6).get();
        metric.update(
                GradTensor.of(new float[] {0.0f}, 1),
                GradTensor.of(new float[] {1.0f}, 1));
        assertEquals(tweedieDeviance(1.0, 1e-6, 1.5), metric.value(), 1e-6);

        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanTweedieDeviance(0.9));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanTweedieDeviance(2.1));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanTweedieDeviance(1.5, true, 0.0));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanTweedieDeviance(1.5, false).get().update(
                GradTensor.of(new float[] {-0.1f}, 1),
                GradTensor.of(new float[] {1.0f}, 1)));
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.meanTweedieDeviance(2.0).get().update(
                GradTensor.of(new float[] {0.0f}, 1),
                GradTensor.of(new float[] {0.0f}, 1)));
    }

    private static double poissonDeviance(double target, double mean) {
        if (target == 0.0) {
            return 2.0 * mean;
        }
        return 2.0 * (target * Math.log(target / mean) - target + mean);
    }

    private static double tweedieDeviance(double target, double mean, double power) {
        double oneMinusPower = 1.0 - power;
        double twoMinusPower = 2.0 - power;
        return 2.0 * (
                Math.pow(target, twoMinusPower) / (oneMinusPower * twoMinusPower)
                        - target * Math.pow(mean, oneMinusPower) / oneMinusPower
                        + Math.pow(mean, twoMinusPower) / twoMinusPower);
    }

    private static double gammaDeviance(double target, double mean) {
        return 2.0 * ((target - mean) / mean - Math.log(target / mean));
    }
}
