package tech.kayys.gollek.ml;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Public regression metric factories inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlRegressionMetricsFacade extends GollekDlMultiLabelMetricsFacade {
    protected GollekDlRegressionMetricsFacade() {
    }

    public static Supplier<CanonicalTrainer.Metric> meanAbsoluteErrorMetric() {
        return CanonicalTrainer.Metrics.meanAbsoluteError();
    }

    public static Supplier<CanonicalTrainer.Metric> maeMetric() {
        return CanonicalTrainer.Metrics.mae();
    }

    public static Supplier<CanonicalTrainer.Metric> meanSquaredErrorMetric() {
        return CanonicalTrainer.Metrics.meanSquaredError();
    }

    public static Supplier<CanonicalTrainer.Metric> mseMetric() {
        return CanonicalTrainer.Metrics.mse();
    }

    public static Supplier<CanonicalTrainer.Metric> rootMeanSquaredErrorMetric() {
        return CanonicalTrainer.Metrics.rootMeanSquaredError();
    }

    public static Supplier<CanonicalTrainer.Metric> rmseMetric() {
        return CanonicalTrainer.Metrics.rmse();
    }

    public static Supplier<CanonicalTrainer.Metric> meanSquaredLogErrorMetric() {
        return CanonicalTrainer.Metrics.meanSquaredLogError();
    }

    public static Supplier<CanonicalTrainer.Metric> msleMetric() {
        return CanonicalTrainer.Metrics.msle();
    }

    public static Supplier<CanonicalTrainer.Metric> rootMeanSquaredLogErrorMetric() {
        return CanonicalTrainer.Metrics.rootMeanSquaredLogError();
    }

    public static Supplier<CanonicalTrainer.Metric> rmsleMetric() {
        return CanonicalTrainer.Metrics.rmsle();
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDevianceMetric() {
        return CanonicalTrainer.Metrics.meanPoissonDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDevianceMetric(boolean logInput) {
        return CanonicalTrainer.Metrics.meanPoissonDeviance(logInput);
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDevianceMetric(boolean logInput, double eps) {
        return CanonicalTrainer.Metrics.meanPoissonDeviance(logInput, eps);
    }

    public static Supplier<CanonicalTrainer.Metric> poissonDevianceMetric() {
        return CanonicalTrainer.Metrics.poissonDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> poissonLogRateDevianceMetric() {
        return CanonicalTrainer.Metrics.poissonLogRateDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDevianceMetric() {
        return CanonicalTrainer.Metrics.meanTweedieDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDevianceMetric(double power) {
        return CanonicalTrainer.Metrics.meanTweedieDeviance(power);
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDevianceMetric(double power, boolean logInput) {
        return CanonicalTrainer.Metrics.meanTweedieDeviance(power, logInput);
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDevianceMetric(
            double power,
            boolean logInput,
            double eps) {
        return CanonicalTrainer.Metrics.meanTweedieDeviance(power, logInput, eps);
    }

    public static Supplier<CanonicalTrainer.Metric> tweedieDevianceMetric() {
        return CanonicalTrainer.Metrics.tweedieDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> compoundPoissonGammaDevianceMetric() {
        return CanonicalTrainer.Metrics.compoundPoissonGammaDeviance();
    }

    public static Supplier<CanonicalTrainer.Metric> medianAbsoluteErrorMetric() {
        return CanonicalTrainer.Metrics.medianAbsoluteError();
    }

    public static Supplier<CanonicalTrainer.Metric> medaeMetric() {
        return CanonicalTrainer.Metrics.medae();
    }

    public static Supplier<CanonicalTrainer.Metric> maxErrorMetric() {
        return CanonicalTrainer.Metrics.maxError();
    }

    public static Supplier<CanonicalTrainer.Metric> pinballLossMetric(double quantile) {
        return CanonicalTrainer.Metrics.pinballLoss(quantile);
    }

    public static Supplier<CanonicalTrainer.Metric> meanPinballLossMetric(double quantile) {
        return CanonicalTrainer.Metrics.meanPinballLoss(quantile);
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalCoverageMetric() {
        return CanonicalTrainer.Metrics.predictionIntervalCoverage();
    }

    public static Supplier<CanonicalTrainer.Metric> picpMetric() {
        return CanonicalTrainer.Metrics.picp();
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalMeanWidthMetric() {
        return CanonicalTrainer.Metrics.predictionIntervalMeanWidth();
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalNormalizedMeanWidthMetric() {
        return CanonicalTrainer.Metrics.predictionIntervalNormalizedMeanWidth();
    }

    public static Supplier<CanonicalTrainer.Metric> r2ScoreMetric() {
        return CanonicalTrainer.Metrics.r2Score();
    }

    public static Supplier<CanonicalTrainer.Metric> r2Metric() {
        return CanonicalTrainer.Metrics.r2();
    }

    public static Supplier<CanonicalTrainer.Metric> meanAbsolutePercentageErrorMetric() {
        return CanonicalTrainer.Metrics.meanAbsolutePercentageError();
    }

    public static Supplier<CanonicalTrainer.Metric> mapeMetric() {
        return CanonicalTrainer.Metrics.mape();
    }

    public static Supplier<CanonicalTrainer.Metric> symmetricMeanAbsolutePercentageErrorMetric() {
        return CanonicalTrainer.Metrics.symmetricMeanAbsolutePercentageError();
    }

    public static Supplier<CanonicalTrainer.Metric> smapeMetric() {
        return CanonicalTrainer.Metrics.smape();
    }

    public static Supplier<CanonicalTrainer.Metric> meanBiasErrorMetric() {
        return CanonicalTrainer.Metrics.meanBiasError();
    }

    public static Supplier<CanonicalTrainer.Metric> mbeMetric() {
        return CanonicalTrainer.Metrics.mbe();
    }

    public static Supplier<CanonicalTrainer.Metric> explainedVarianceMetric() {
        return CanonicalTrainer.Metrics.explainedVariance();
    }

    public static Supplier<CanonicalTrainer.Metric> explainedVarianceScoreMetric() {
        return CanonicalTrainer.Metrics.explainedVarianceScore();
    }
}
