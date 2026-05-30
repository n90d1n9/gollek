package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Legacy metric factory catalog inherited by {@link CanonicalTrainer.Metrics}.
 */
@SuppressWarnings("deprecation")
abstract class CanonicalTrainerMetricFactories {

    public static Supplier<CanonicalTrainer.Metric> classificationAccuracy() {
        return legacy(TrainingMetrics.classificationAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> accuracy() {
        return legacy(TrainingMetrics.accuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationConfusionMatrix() {
        return legacy(TrainingMetrics.classificationConfusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> confusionMatrix() {
        return legacy(TrainingMetrics.confusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> topKAccuracy(int k) {
        return legacy(TrainingMetrics.topKAccuracy(k));
    }

    public static Supplier<CanonicalTrainer.Metric> classificationLogLoss() {
        return legacy(TrainingMetrics.classificationLogLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationCrossEntropy() {
        return legacy(TrainingMetrics.classificationCrossEntropy());
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingTokenAccuracy() {
        return legacy(TrainingMetrics.causalLanguageModelingTokenAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingTokenAccuracy(float ignoreIndex) {
        return legacy(TrainingMetrics.causalLanguageModelingTokenAccuracy(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenAccuracy() {
        return legacy(TrainingMetrics.nextTokenAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenAccuracy(float ignoreIndex) {
        return legacy(TrainingMetrics.nextTokenAccuracy(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingLogLoss() {
        return legacy(TrainingMetrics.causalLanguageModelingLogLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingLogLoss(float ignoreIndex) {
        return legacy(TrainingMetrics.causalLanguageModelingLogLoss(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingCrossEntropy() {
        return legacy(TrainingMetrics.causalLanguageModelingCrossEntropy());
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingCrossEntropy(float ignoreIndex) {
        return legacy(TrainingMetrics.causalLanguageModelingCrossEntropy(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingPerplexity() {
        return legacy(TrainingMetrics.causalLanguageModelingPerplexity());
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingPerplexity(float ignoreIndex) {
        return legacy(TrainingMetrics.causalLanguageModelingPerplexity(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenPerplexity() {
        return legacy(TrainingMetrics.nextTokenPerplexity());
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenPerplexity(float ignoreIndex) {
        return legacy(TrainingMetrics.nextTokenPerplexity(ignoreIndex));
    }

    public static Supplier<CanonicalTrainer.Metric> classificationBalancedAccuracy() {
        return legacy(TrainingMetrics.classificationBalancedAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> balancedAccuracy() {
        return legacy(TrainingMetrics.balancedAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMatthewsCorrelationCoefficient() {
        return legacy(TrainingMetrics.classificationMatthewsCorrelationCoefficient());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMcc() {
        return legacy(TrainingMetrics.classificationMcc());
    }

    public static Supplier<CanonicalTrainer.Metric> matthewsCorrelationCoefficient() {
        return legacy(TrainingMetrics.matthewsCorrelationCoefficient());
    }

    public static Supplier<CanonicalTrainer.Metric> mcc() {
        return legacy(TrainingMetrics.mcc());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedPrecision() {
        return legacy(TrainingMetrics.classificationWeightedPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedRecall() {
        return legacy(TrainingMetrics.classificationWeightedRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedF1() {
        return legacy(TrainingMetrics.classificationWeightedF1());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationCohensKappa() {
        return legacy(TrainingMetrics.classificationCohensKappa());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationKappa() {
        return legacy(TrainingMetrics.classificationKappa());
    }

    public static Supplier<CanonicalTrainer.Metric> cohensKappa() {
        return legacy(TrainingMetrics.cohensKappa());
    }

    public static Supplier<CanonicalTrainer.Metric> kappa() {
        return legacy(TrainingMetrics.kappa());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationBrierScore() {
        return legacy(TrainingMetrics.classificationBrierScore());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationExpectedCalibrationError() {
        return legacy(TrainingMetrics.classificationExpectedCalibrationError());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationExpectedCalibrationError(int bins) {
        return legacy(TrainingMetrics.classificationExpectedCalibrationError(bins));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracy() {
        return legacy(TrainingMetrics.binaryAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracy(float logitThreshold) {
        return legacy(TrainingMetrics.binaryAccuracy(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBalancedAccuracy() {
        return legacy(TrainingMetrics.binaryBalancedAccuracy());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBalancedAccuracy(float logitThreshold) {
        return legacy(TrainingMetrics.binaryBalancedAccuracy(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMatthewsCorrelationCoefficient() {
        return legacy(TrainingMetrics.binaryMatthewsCorrelationCoefficient());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMatthewsCorrelationCoefficient(float logitThreshold) {
        return legacy(TrainingMetrics.binaryMatthewsCorrelationCoefficient(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMcc() {
        return legacy(TrainingMetrics.binaryMcc());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMcc(float logitThreshold) {
        return legacy(TrainingMetrics.binaryMcc(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCohensKappa() {
        return legacy(TrainingMetrics.binaryCohensKappa());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCohensKappa(float logitThreshold) {
        return legacy(TrainingMetrics.binaryCohensKappa(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryKappa() {
        return legacy(TrainingMetrics.binaryKappa());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryKappa(float logitThreshold) {
        return legacy(TrainingMetrics.binaryKappa(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrix() {
        return legacy(TrainingMetrics.binaryConfusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrix(float logitThreshold) {
        return legacy(TrainingMetrics.binaryConfusionMatrix(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecision() {
        return legacy(TrainingMetrics.binaryPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecision(float logitThreshold) {
        return legacy(TrainingMetrics.binaryPrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecall() {
        return legacy(TrainingMetrics.binaryRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecall(float logitThreshold) {
        return legacy(TrainingMetrics.binaryRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1() {
        return legacy(TrainingMetrics.binaryF1());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1(float logitThreshold) {
        return legacy(TrainingMetrics.binaryF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRocAuc() {
        return legacy(TrainingMetrics.binaryRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAuroc() {
        return legacy(TrainingMetrics.binaryAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAveragePrecision() {
        return legacy(TrainingMetrics.binaryAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBestF1() {
        return legacy(TrainingMetrics.binaryBestF1());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBestF1Threshold() {
        return legacy(TrainingMetrics.binaryBestF1Threshold());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecisionAtRecall(double minimumRecall) {
        return legacy(TrainingMetrics.binaryPrecisionAtRecall(minimumRecall));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecallAtPrecision(double minimumPrecision) {
        return legacy(TrainingMetrics.binaryRecallAtPrecision(minimumPrecision));
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBrierScore() {
        return legacy(TrainingMetrics.binaryBrierScore());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryLogLoss() {
        return legacy(TrainingMetrics.binaryLogLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCrossEntropy() {
        return legacy(TrainingMetrics.binaryCrossEntropy());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryExpectedCalibrationError() {
        return legacy(TrainingMetrics.binaryExpectedCalibrationError());
    }

    public static Supplier<CanonicalTrainer.Metric> binaryExpectedCalibrationError(int bins) {
        return legacy(TrainingMetrics.binaryExpectedCalibrationError(bins));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatch() {
        return legacy(TrainingMetrics.multiLabelExactMatch());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatch(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelExactMatch(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLoss() {
        return legacy(TrainingMetrics.multiLabelHammingLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLoss(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelHammingLoss(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelConfusionMatrix() {
        return legacy(TrainingMetrics.multiLabelConfusionMatrix());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelConfusionMatrix(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelConfusionMatrix(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroPrecision() {
        return legacy(TrainingMetrics.multiLabelMicroPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroPrecision(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMicroPrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroRecall() {
        return legacy(TrainingMetrics.multiLabelMicroRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroRecall(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMicroRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroF1() {
        return legacy(TrainingMetrics.multiLabelMicroF1());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroF1(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMicroF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSamplePrecision() {
        return legacy(TrainingMetrics.multiLabelSamplePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSamplePrecision(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelSamplePrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleRecall() {
        return legacy(TrainingMetrics.multiLabelSampleRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleRecall(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelSampleRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleF1() {
        return legacy(TrainingMetrics.multiLabelSampleF1());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleF1(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelSampleF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleJaccard() {
        return legacy(TrainingMetrics.multiLabelSampleJaccard());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleJaccard(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelSampleJaccard(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecision() {
        return legacy(TrainingMetrics.multiLabelMacroPrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecision(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroPrecision(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecall() {
        return legacy(TrainingMetrics.multiLabelMacroRecall());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecall(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroRecall(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1() {
        return legacy(TrainingMetrics.multiLabelMacroF1());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1(float logitThreshold) {
        return legacy(TrainingMetrics.multiLabelMacroF1(logitThreshold));
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRocAuc() {
        return legacy(TrainingMetrics.multiLabelMacroRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAuroc() {
        return legacy(TrainingMetrics.multiLabelMacroAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAveragePrecision() {
        return legacy(TrainingMetrics.multiLabelMacroAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelLabelRankingAveragePrecision() {
        return legacy(TrainingMetrics.multiLabelLabelRankingAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelLrap() {
        return legacy(TrainingMetrics.multiLabelLrap());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelRankingLoss() {
        return legacy(TrainingMetrics.multiLabelRankingLoss());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelCoverageError() {
        return legacy(TrainingMetrics.multiLabelCoverageError());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroBestF1() {
        return legacy(TrainingMetrics.multiLabelMacroBestF1());
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelBestF1Thresholds() {
        return legacy(TrainingMetrics.multiLabelBestF1Thresholds());
    }

    public static Supplier<CanonicalTrainer.Metric> precision() {
        return legacy(TrainingMetrics.precision());
    }

    public static Supplier<CanonicalTrainer.Metric> recall() {
        return legacy(TrainingMetrics.recall());
    }

    public static Supplier<CanonicalTrainer.Metric> f1() {
        return legacy(TrainingMetrics.f1());
    }

    public static Supplier<CanonicalTrainer.Metric> macroF1() {
        return legacy(TrainingMetrics.macroF1());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroRocAuc() {
        return legacy(TrainingMetrics.classificationMacroRocAuc());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAuroc() {
        return legacy(TrainingMetrics.classificationMacroAuroc());
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAveragePrecision() {
        return legacy(TrainingMetrics.classificationMacroAveragePrecision());
    }

    public static Supplier<CanonicalTrainer.Metric> meanAbsoluteError() {
        return legacy(TrainingMetrics.meanAbsoluteError());
    }

    public static Supplier<CanonicalTrainer.Metric> mae() {
        return legacy(TrainingMetrics.mae());
    }

    public static Supplier<CanonicalTrainer.Metric> meanSquaredError() {
        return legacy(TrainingMetrics.meanSquaredError());
    }

    public static Supplier<CanonicalTrainer.Metric> mse() {
        return legacy(TrainingMetrics.mse());
    }

    public static Supplier<CanonicalTrainer.Metric> rootMeanSquaredError() {
        return legacy(TrainingMetrics.rootMeanSquaredError());
    }

    public static Supplier<CanonicalTrainer.Metric> rmse() {
        return legacy(TrainingMetrics.rmse());
    }

    public static Supplier<CanonicalTrainer.Metric> meanSquaredLogError() {
        return legacy(TrainingMetrics.meanSquaredLogError());
    }

    public static Supplier<CanonicalTrainer.Metric> msle() {
        return legacy(TrainingMetrics.msle());
    }

    public static Supplier<CanonicalTrainer.Metric> rootMeanSquaredLogError() {
        return legacy(TrainingMetrics.rootMeanSquaredLogError());
    }

    public static Supplier<CanonicalTrainer.Metric> rmsle() {
        return legacy(TrainingMetrics.rmsle());
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDeviance() {
        return legacy(TrainingMetrics.meanPoissonDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDeviance(boolean logInput) {
        return legacy(TrainingMetrics.meanPoissonDeviance(logInput));
    }

    public static Supplier<CanonicalTrainer.Metric> meanPoissonDeviance(boolean logInput, double eps) {
        return legacy(TrainingMetrics.meanPoissonDeviance(logInput, eps));
    }

    public static Supplier<CanonicalTrainer.Metric> poissonDeviance() {
        return legacy(TrainingMetrics.poissonDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> poissonLogRateDeviance() {
        return legacy(TrainingMetrics.poissonLogRateDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDeviance() {
        return legacy(TrainingMetrics.meanTweedieDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDeviance(double power) {
        return legacy(TrainingMetrics.meanTweedieDeviance(power));
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDeviance(double power, boolean logInput) {
        return legacy(TrainingMetrics.meanTweedieDeviance(power, logInput));
    }

    public static Supplier<CanonicalTrainer.Metric> meanTweedieDeviance(
            double power,
            boolean logInput,
            double eps) {
        return legacy(TrainingMetrics.meanTweedieDeviance(power, logInput, eps));
    }

    public static Supplier<CanonicalTrainer.Metric> tweedieDeviance() {
        return legacy(TrainingMetrics.tweedieDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> compoundPoissonGammaDeviance() {
        return legacy(TrainingMetrics.compoundPoissonGammaDeviance());
    }

    public static Supplier<CanonicalTrainer.Metric> medianAbsoluteError() {
        return legacy(TrainingMetrics.medianAbsoluteError());
    }

    public static Supplier<CanonicalTrainer.Metric> medae() {
        return legacy(TrainingMetrics.medae());
    }

    public static Supplier<CanonicalTrainer.Metric> maxError() {
        return legacy(TrainingMetrics.maxError());
    }

    public static Supplier<CanonicalTrainer.Metric> pinballLoss(double quantile) {
        return legacy(TrainingMetrics.pinballLoss(quantile));
    }

    public static Supplier<CanonicalTrainer.Metric> meanPinballLoss(double quantile) {
        return legacy(TrainingMetrics.meanPinballLoss(quantile));
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalCoverage() {
        return legacy(TrainingMetrics.predictionIntervalCoverage());
    }

    public static Supplier<CanonicalTrainer.Metric> picp() {
        return legacy(TrainingMetrics.picp());
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalMeanWidth() {
        return legacy(TrainingMetrics.predictionIntervalMeanWidth());
    }

    public static Supplier<CanonicalTrainer.Metric> predictionIntervalNormalizedMeanWidth() {
        return legacy(TrainingMetrics.predictionIntervalNormalizedMeanWidth());
    }

    public static Supplier<CanonicalTrainer.Metric> r2Score() {
        return legacy(TrainingMetrics.r2Score());
    }

    public static Supplier<CanonicalTrainer.Metric> r2() {
        return legacy(TrainingMetrics.r2());
    }

    public static Supplier<CanonicalTrainer.Metric> meanAbsolutePercentageError() {
        return legacy(TrainingMetrics.meanAbsolutePercentageError());
    }

    public static Supplier<CanonicalTrainer.Metric> mape() {
        return legacy(TrainingMetrics.mape());
    }

    public static Supplier<CanonicalTrainer.Metric> symmetricMeanAbsolutePercentageError() {
        return legacy(TrainingMetrics.symmetricMeanAbsolutePercentageError());
    }

    public static Supplier<CanonicalTrainer.Metric> smape() {
        return legacy(TrainingMetrics.smape());
    }

    public static Supplier<CanonicalTrainer.Metric> meanBiasError() {
        return legacy(TrainingMetrics.meanBiasError());
    }

    public static Supplier<CanonicalTrainer.Metric> mbe() {
        return legacy(TrainingMetrics.mbe());
    }

    public static Supplier<CanonicalTrainer.Metric> explainedVariance() {
        return legacy(TrainingMetrics.explainedVariance());
    }

    public static Supplier<CanonicalTrainer.Metric> explainedVarianceScore() {
        return legacy(TrainingMetrics.explainedVarianceScore());
    }

    private static Supplier<CanonicalTrainer.Metric> legacy(Supplier<? extends TrainingMetric> factory) {
        return TrainerLegacyMetrics.legacy(factory);
    }
}
