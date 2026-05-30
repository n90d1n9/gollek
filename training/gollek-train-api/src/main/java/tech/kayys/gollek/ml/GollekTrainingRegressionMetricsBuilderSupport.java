package tech.kayys.gollek.ml;

/**
 * Fluent regression metric registration for training option builders.
 */
public abstract class GollekTrainingRegressionMetricsBuilderSupport<
        B extends GollekTrainingRegressionMetricsBuilderSupport<B>>
        extends GollekTrainingMultiLabelMetricsBuilderSupport<B> {

    public B meanAbsoluteErrorMetric() {
        return metric(Gollek.DL.meanAbsoluteErrorMetric());
    }

    public B maeMetric() {
        return meanAbsoluteErrorMetric();
    }

    public B meanSquaredErrorMetric() {
        return metric(Gollek.DL.meanSquaredErrorMetric());
    }

    public B mseMetric() {
        return meanSquaredErrorMetric();
    }

    public B rootMeanSquaredErrorMetric() {
        return metric(Gollek.DL.rootMeanSquaredErrorMetric());
    }

    public B rmseMetric() {
        return rootMeanSquaredErrorMetric();
    }

    public B meanSquaredLogErrorMetric() {
        return metric(Gollek.DL.meanSquaredLogErrorMetric());
    }

    public B msleMetric() {
        return meanSquaredLogErrorMetric();
    }

    public B rootMeanSquaredLogErrorMetric() {
        return metric(Gollek.DL.rootMeanSquaredLogErrorMetric());
    }

    public B rmsleMetric() {
        return rootMeanSquaredLogErrorMetric();
    }

    public B meanPoissonDevianceMetric() {
        return metric(Gollek.DL.meanPoissonDevianceMetric());
    }

    public B meanPoissonDevianceMetric(boolean logInput) {
        return metric(Gollek.DL.meanPoissonDevianceMetric(logInput));
    }

    public B meanPoissonDevianceMetric(boolean logInput, double eps) {
        return metric(Gollek.DL.meanPoissonDevianceMetric(logInput, eps));
    }

    public B poissonDevianceMetric() {
        return meanPoissonDevianceMetric();
    }

    public B poissonLogRateDevianceMetric() {
        return metric(Gollek.DL.poissonLogRateDevianceMetric());
    }

    public B meanTweedieDevianceMetric() {
        return metric(Gollek.DL.meanTweedieDevianceMetric());
    }

    public B meanTweedieDevianceMetric(double power) {
        return metric(Gollek.DL.meanTweedieDevianceMetric(power));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput) {
        return metric(Gollek.DL.meanTweedieDevianceMetric(power, logInput));
    }

    public B meanTweedieDevianceMetric(double power, boolean logInput, double eps) {
        return metric(Gollek.DL.meanTweedieDevianceMetric(power, logInput, eps));
    }

    public B tweedieDevianceMetric() {
        return meanTweedieDevianceMetric();
    }

    public B compoundPoissonGammaDevianceMetric() {
        return metric(Gollek.DL.compoundPoissonGammaDevianceMetric());
    }

    public B medianAbsoluteErrorMetric() {
        return metric(Gollek.DL.medianAbsoluteErrorMetric());
    }

    public B medaeMetric() {
        return medianAbsoluteErrorMetric();
    }

    public B maxErrorMetric() {
        return metric(Gollek.DL.maxErrorMetric());
    }

    public B pinballLossMetric(double quantile) {
        return metric(Gollek.DL.pinballLossMetric(quantile));
    }

    public B meanPinballLossMetric(double quantile) {
        return metric(Gollek.DL.meanPinballLossMetric(quantile));
    }

    public B predictionIntervalCoverageMetric() {
        return metric(Gollek.DL.predictionIntervalCoverageMetric());
    }

    public B picpMetric() {
        return predictionIntervalCoverageMetric();
    }

    public B predictionIntervalMeanWidthMetric() {
        return metric(Gollek.DL.predictionIntervalMeanWidthMetric());
    }

    public B predictionIntervalNormalizedMeanWidthMetric() {
        return metric(Gollek.DL.predictionIntervalNormalizedMeanWidthMetric());
    }

    public B r2ScoreMetric() {
        return metric(Gollek.DL.r2ScoreMetric());
    }

    public B r2Metric() {
        return r2ScoreMetric();
    }

    public B meanAbsolutePercentageErrorMetric() {
        return metric(Gollek.DL.meanAbsolutePercentageErrorMetric());
    }

    public B mapeMetric() {
        return meanAbsolutePercentageErrorMetric();
    }

    public B symmetricMeanAbsolutePercentageErrorMetric() {
        return metric(Gollek.DL.symmetricMeanAbsolutePercentageErrorMetric());
    }

    public B smapeMetric() {
        return symmetricMeanAbsolutePercentageErrorMetric();
    }

    public B meanBiasErrorMetric() {
        return metric(Gollek.DL.meanBiasErrorMetric());
    }

    public B mbeMetric() {
        return meanBiasErrorMetric();
    }

    public B explainedVarianceMetric() {
        return metric(Gollek.DL.explainedVarianceMetric());
    }

    public B explainedVarianceScoreMetric() {
        return explainedVarianceMetric();
    }

    public B regressionMetrics() {
        return meanAbsoluteErrorMetric()
                .meanSquaredErrorMetric()
                .rootMeanSquaredErrorMetric()
                .r2ScoreMetric();
    }

    public B regressionExtendedMetrics() {
        return regressionMetrics()
                .medianAbsoluteErrorMetric()
                .maxErrorMetric()
                .meanAbsolutePercentageErrorMetric()
                .symmetricMeanAbsolutePercentageErrorMetric()
                .meanBiasErrorMetric()
                .explainedVarianceMetric();
    }

    public B regressionLogScaleMetrics() {
        return meanSquaredLogErrorMetric()
                .rootMeanSquaredLogErrorMetric();
    }

    public B countRegressionMetrics() {
        return poissonLogRateDevianceMetric()
                .compoundPoissonGammaDevianceMetric();
    }

    public B regressionQuantileMetrics() {
        return pinballLossMetric(0.1)
                .pinballLossMetric(0.5)
                .pinballLossMetric(0.9);
    }

    public B regressionPredictionIntervalMetrics() {
        return predictionIntervalCoverageMetric()
                .predictionIntervalMeanWidthMetric()
                .predictionIntervalNormalizedMeanWidthMetric();
    }
}
