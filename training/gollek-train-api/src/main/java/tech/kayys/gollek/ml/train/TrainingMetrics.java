package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Built-in metric factories decoupled from the trainer type.
 */
public final class TrainingMetrics {
    private TrainingMetrics() {
    }

    public static <S> TrainingMetricBuilder<S> custom(String name, Supplier<S> stateFactory) {
        return new TrainingMetricBuilder<>(name, stateFactory);
    }

    public static Supplier<TrainingMetric> classificationAccuracy() {
        return BuiltInTrainingMetrics.classificationAccuracy();
    }

    public static Supplier<TrainingMetric> accuracy() {
        return classificationAccuracy();
    }

    public static Supplier<TrainingMetric> classificationConfusionMatrix() {
        return BuiltInTrainingMetrics.classificationConfusionMatrix();
    }

    public static Supplier<TrainingMetric> confusionMatrix() {
        return classificationConfusionMatrix();
    }

    public static Supplier<TrainingMetric> topKAccuracy(int k) {
        return BuiltInTrainingMetrics.topKAccuracy(k);
    }

    public static Supplier<TrainingMetric> classificationLogLoss() {
        return BuiltInTrainingMetrics.classificationLogLoss();
    }

    public static Supplier<TrainingMetric> classificationCrossEntropy() {
        return BuiltInTrainingMetrics.classificationCrossEntropy();
    }

    public static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy() {
        return BuiltInTrainingMetrics.causalLanguageModelingTokenAccuracy();
    }

    public static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy(float ignoreIndex) {
        return BuiltInTrainingMetrics.causalLanguageModelingTokenAccuracy(ignoreIndex);
    }

    public static Supplier<TrainingMetric> nextTokenAccuracy() {
        return causalLanguageModelingTokenAccuracy();
    }

    public static Supplier<TrainingMetric> nextTokenAccuracy(float ignoreIndex) {
        return causalLanguageModelingTokenAccuracy(ignoreIndex);
    }

    public static Supplier<TrainingMetric> causalLanguageModelingLogLoss() {
        return BuiltInTrainingMetrics.causalLanguageModelingLogLoss();
    }

    public static Supplier<TrainingMetric> causalLanguageModelingLogLoss(float ignoreIndex) {
        return BuiltInTrainingMetrics.causalLanguageModelingLogLoss(ignoreIndex);
    }

    public static Supplier<TrainingMetric> causalLanguageModelingCrossEntropy() {
        return causalLanguageModelingLogLoss();
    }

    public static Supplier<TrainingMetric> causalLanguageModelingCrossEntropy(float ignoreIndex) {
        return causalLanguageModelingLogLoss(ignoreIndex);
    }

    public static Supplier<TrainingMetric> causalLanguageModelingPerplexity() {
        return BuiltInTrainingMetrics.causalLanguageModelingPerplexity();
    }

    public static Supplier<TrainingMetric> causalLanguageModelingPerplexity(float ignoreIndex) {
        return BuiltInTrainingMetrics.causalLanguageModelingPerplexity(ignoreIndex);
    }

    public static Supplier<TrainingMetric> nextTokenPerplexity() {
        return causalLanguageModelingPerplexity();
    }

    public static Supplier<TrainingMetric> nextTokenPerplexity(float ignoreIndex) {
        return causalLanguageModelingPerplexity(ignoreIndex);
    }

    public static Supplier<TrainingMetric> classificationBalancedAccuracy() {
        return BuiltInTrainingMetrics.classificationBalancedAccuracy();
    }

    public static Supplier<TrainingMetric> balancedAccuracy() {
        return classificationBalancedAccuracy();
    }

    public static Supplier<TrainingMetric> classificationMatthewsCorrelationCoefficient() {
        return BuiltInTrainingMetrics.classificationMatthewsCorrelationCoefficient();
    }

    public static Supplier<TrainingMetric> classificationMcc() {
        return classificationMatthewsCorrelationCoefficient();
    }

    public static Supplier<TrainingMetric> matthewsCorrelationCoefficient() {
        return classificationMatthewsCorrelationCoefficient();
    }

    public static Supplier<TrainingMetric> mcc() {
        return classificationMatthewsCorrelationCoefficient();
    }

    public static Supplier<TrainingMetric> classificationWeightedPrecision() {
        return BuiltInTrainingMetrics.classificationWeightedPrecision();
    }

    public static Supplier<TrainingMetric> classificationWeightedRecall() {
        return BuiltInTrainingMetrics.classificationWeightedRecall();
    }

    public static Supplier<TrainingMetric> classificationWeightedF1() {
        return BuiltInTrainingMetrics.classificationWeightedF1();
    }

    public static Supplier<TrainingMetric> classificationCohensKappa() {
        return BuiltInTrainingMetrics.classificationCohensKappa();
    }

    public static Supplier<TrainingMetric> classificationKappa() {
        return classificationCohensKappa();
    }

    public static Supplier<TrainingMetric> cohensKappa() {
        return classificationCohensKappa();
    }

    public static Supplier<TrainingMetric> kappa() {
        return classificationCohensKappa();
    }

    public static Supplier<TrainingMetric> classificationBrierScore() {
        return BuiltInTrainingMetrics.classificationBrierScore();
    }

    public static Supplier<TrainingMetric> classificationExpectedCalibrationError() {
        return BuiltInTrainingMetrics.classificationExpectedCalibrationError();
    }

    public static Supplier<TrainingMetric> classificationExpectedCalibrationError(int bins) {
        return BuiltInTrainingMetrics.classificationExpectedCalibrationError(bins);
    }

    public static Supplier<TrainingMetric> binaryAccuracy() {
        return BuiltInTrainingMetrics.binaryAccuracy();
    }

    public static Supplier<TrainingMetric> binaryAccuracy(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryAccuracy(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryBalancedAccuracy() {
        return BuiltInTrainingMetrics.binaryBalancedAccuracy();
    }

    public static Supplier<TrainingMetric> binaryBalancedAccuracy(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryBalancedAccuracy(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient() {
        return BuiltInTrainingMetrics.binaryMatthewsCorrelationCoefficient();
    }

    public static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryMatthewsCorrelationCoefficient(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryMcc() {
        return BuiltInTrainingMetrics.binaryMcc();
    }

    public static Supplier<TrainingMetric> binaryMcc(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryMcc(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryCohensKappa() {
        return BuiltInTrainingMetrics.binaryCohensKappa();
    }

    public static Supplier<TrainingMetric> binaryCohensKappa(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryCohensKappa(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryKappa() {
        return BuiltInTrainingMetrics.binaryKappa();
    }

    public static Supplier<TrainingMetric> binaryKappa(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryKappa(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryConfusionMatrix() {
        return BuiltInTrainingMetrics.binaryConfusionMatrix();
    }

    public static Supplier<TrainingMetric> binaryConfusionMatrix(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryConfusionMatrix(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryPrecision() {
        return BuiltInTrainingMetrics.binaryPrecision();
    }

    public static Supplier<TrainingMetric> binaryPrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryPrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryRecall() {
        return BuiltInTrainingMetrics.binaryRecall();
    }

    public static Supplier<TrainingMetric> binaryRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryF1() {
        return BuiltInTrainingMetrics.binaryF1();
    }

    public static Supplier<TrainingMetric> binaryF1(float logitThreshold) {
        return BuiltInTrainingMetrics.binaryF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> binaryRocAuc() {
        return BuiltInTrainingMetrics.binaryRocAuc();
    }

    public static Supplier<TrainingMetric> binaryAuroc() {
        return binaryRocAuc();
    }

    public static Supplier<TrainingMetric> binaryAveragePrecision() {
        return BuiltInTrainingMetrics.binaryAveragePrecision();
    }

    public static Supplier<TrainingMetric> binaryBestF1() {
        return BuiltInTrainingMetrics.binaryBestF1();
    }

    public static Supplier<TrainingMetric> binaryBestF1Threshold() {
        return BuiltInTrainingMetrics.binaryBestF1Threshold();
    }

    public static Supplier<TrainingMetric> binaryPrecisionAtRecall(double minimumRecall) {
        return BuiltInTrainingMetrics.binaryPrecisionAtRecall(minimumRecall);
    }

    public static Supplier<TrainingMetric> binaryRecallAtPrecision(double minimumPrecision) {
        return BuiltInTrainingMetrics.binaryRecallAtPrecision(minimumPrecision);
    }

    public static Supplier<TrainingMetric> binaryBrierScore() {
        return BuiltInTrainingMetrics.binaryBrierScore();
    }

    public static Supplier<TrainingMetric> binaryLogLoss() {
        return BuiltInTrainingMetrics.binaryLogLoss();
    }

    public static Supplier<TrainingMetric> binaryCrossEntropy() {
        return BuiltInTrainingMetrics.binaryCrossEntropy();
    }

    public static Supplier<TrainingMetric> binaryExpectedCalibrationError() {
        return BuiltInTrainingMetrics.binaryExpectedCalibrationError();
    }

    public static Supplier<TrainingMetric> binaryExpectedCalibrationError(int bins) {
        return BuiltInTrainingMetrics.binaryExpectedCalibrationError(bins);
    }

    public static Supplier<TrainingMetric> multiLabelExactMatch() {
        return BuiltInTrainingMetrics.multiLabelExactMatch();
    }

    public static Supplier<TrainingMetric> multiLabelExactMatch(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelExactMatch(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelHammingLoss() {
        return BuiltInTrainingMetrics.multiLabelHammingLoss();
    }

    public static Supplier<TrainingMetric> multiLabelHammingLoss(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelHammingLoss(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelConfusionMatrix() {
        return BuiltInTrainingMetrics.multiLabelConfusionMatrix();
    }

    public static Supplier<TrainingMetric> multiLabelConfusionMatrix(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelConfusionMatrix(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMicroPrecision() {
        return BuiltInTrainingMetrics.multiLabelMicroPrecision();
    }

    public static Supplier<TrainingMetric> multiLabelMicroPrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMicroPrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMicroRecall() {
        return BuiltInTrainingMetrics.multiLabelMicroRecall();
    }

    public static Supplier<TrainingMetric> multiLabelMicroRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMicroRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMicroF1() {
        return BuiltInTrainingMetrics.multiLabelMicroF1();
    }

    public static Supplier<TrainingMetric> multiLabelMicroF1(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMicroF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelSamplePrecision() {
        return BuiltInTrainingMetrics.multiLabelSamplePrecision();
    }

    public static Supplier<TrainingMetric> multiLabelSamplePrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelSamplePrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelSampleRecall() {
        return BuiltInTrainingMetrics.multiLabelSampleRecall();
    }

    public static Supplier<TrainingMetric> multiLabelSampleRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelSampleRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelSampleF1() {
        return BuiltInTrainingMetrics.multiLabelSampleF1();
    }

    public static Supplier<TrainingMetric> multiLabelSampleF1(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelSampleF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelSampleJaccard() {
        return BuiltInTrainingMetrics.multiLabelSampleJaccard();
    }

    public static Supplier<TrainingMetric> multiLabelSampleJaccard(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelSampleJaccard(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroPrecision() {
        return BuiltInTrainingMetrics.multiLabelMacroPrecision();
    }

    public static Supplier<TrainingMetric> multiLabelMacroPrecision(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroPrecision(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroRecall() {
        return BuiltInTrainingMetrics.multiLabelMacroRecall();
    }

    public static Supplier<TrainingMetric> multiLabelMacroRecall(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroRecall(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroF1() {
        return BuiltInTrainingMetrics.multiLabelMacroF1();
    }

    public static Supplier<TrainingMetric> multiLabelMacroF1(float logitThreshold) {
        return BuiltInTrainingMetrics.multiLabelMacroF1(logitThreshold);
    }

    public static Supplier<TrainingMetric> multiLabelMacroRocAuc() {
        return BuiltInTrainingMetrics.multiLabelMacroRocAuc();
    }

    public static Supplier<TrainingMetric> multiLabelMacroAuroc() {
        return multiLabelMacroRocAuc();
    }

    public static Supplier<TrainingMetric> multiLabelMacroAveragePrecision() {
        return BuiltInTrainingMetrics.multiLabelMacroAveragePrecision();
    }

    public static Supplier<TrainingMetric> multiLabelLabelRankingAveragePrecision() {
        return BuiltInTrainingMetrics.multiLabelLabelRankingAveragePrecision();
    }

    public static Supplier<TrainingMetric> multiLabelLrap() {
        return BuiltInTrainingMetrics.multiLabelLrap();
    }

    public static Supplier<TrainingMetric> multiLabelRankingLoss() {
        return BuiltInTrainingMetrics.multiLabelRankingLoss();
    }

    public static Supplier<TrainingMetric> multiLabelCoverageError() {
        return BuiltInTrainingMetrics.multiLabelCoverageError();
    }

    public static Supplier<TrainingMetric> multiLabelMacroBestF1() {
        return BuiltInTrainingMetrics.multiLabelMacroBestF1();
    }

    public static Supplier<TrainingMetric> multiLabelBestF1Thresholds() {
        return BuiltInTrainingMetrics.multiLabelBestF1Thresholds();
    }

    public static Supplier<TrainingMetric> precision() {
        return BuiltInTrainingMetrics.precision();
    }

    public static Supplier<TrainingMetric> recall() {
        return BuiltInTrainingMetrics.recall();
    }

    public static Supplier<TrainingMetric> f1() {
        return BuiltInTrainingMetrics.f1();
    }

    public static Supplier<TrainingMetric> macroF1() {
        return f1();
    }

    public static Supplier<TrainingMetric> classificationMacroRocAuc() {
        return BuiltInTrainingMetrics.classificationMacroRocAuc();
    }

    public static Supplier<TrainingMetric> classificationMacroAuroc() {
        return classificationMacroRocAuc();
    }

    public static Supplier<TrainingMetric> classificationMacroAveragePrecision() {
        return BuiltInTrainingMetrics.classificationMacroAveragePrecision();
    }

    public static Supplier<TrainingMetric> meanAbsoluteError() {
        return BuiltInTrainingMetrics.meanAbsoluteError();
    }

    public static Supplier<TrainingMetric> mae() {
        return meanAbsoluteError();
    }

    public static Supplier<TrainingMetric> meanSquaredError() {
        return BuiltInTrainingMetrics.meanSquaredError();
    }

    public static Supplier<TrainingMetric> mse() {
        return meanSquaredError();
    }

    public static Supplier<TrainingMetric> rootMeanSquaredError() {
        return BuiltInTrainingMetrics.rootMeanSquaredError();
    }

    public static Supplier<TrainingMetric> rmse() {
        return rootMeanSquaredError();
    }

    public static Supplier<TrainingMetric> meanSquaredLogError() {
        return BuiltInTrainingMetrics.meanSquaredLogError();
    }

    public static Supplier<TrainingMetric> msle() {
        return meanSquaredLogError();
    }

    public static Supplier<TrainingMetric> rootMeanSquaredLogError() {
        return BuiltInTrainingMetrics.rootMeanSquaredLogError();
    }

    public static Supplier<TrainingMetric> rmsle() {
        return rootMeanSquaredLogError();
    }

    public static Supplier<TrainingMetric> meanPoissonDeviance() {
        return BuiltInTrainingMetrics.meanPoissonDeviance();
    }

    public static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput) {
        return BuiltInTrainingMetrics.meanPoissonDeviance(logInput);
    }

    public static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput, double eps) {
        return BuiltInTrainingMetrics.meanPoissonDeviance(logInput, eps);
    }

    public static Supplier<TrainingMetric> poissonDeviance() {
        return meanPoissonDeviance();
    }

    public static Supplier<TrainingMetric> poissonLogRateDeviance() {
        return meanPoissonDeviance(true);
    }

    public static Supplier<TrainingMetric> meanTweedieDeviance() {
        return BuiltInTrainingMetrics.meanTweedieDeviance();
    }

    public static Supplier<TrainingMetric> meanTweedieDeviance(double power) {
        return BuiltInTrainingMetrics.meanTweedieDeviance(power);
    }

    public static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput) {
        return BuiltInTrainingMetrics.meanTweedieDeviance(power, logInput);
    }

    public static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput, double eps) {
        return BuiltInTrainingMetrics.meanTweedieDeviance(power, logInput, eps);
    }

    public static Supplier<TrainingMetric> tweedieDeviance() {
        return meanTweedieDeviance();
    }

    public static Supplier<TrainingMetric> compoundPoissonGammaDeviance() {
        return meanTweedieDeviance();
    }

    public static Supplier<TrainingMetric> medianAbsoluteError() {
        return BuiltInTrainingMetrics.medianAbsoluteError();
    }

    public static Supplier<TrainingMetric> medae() {
        return medianAbsoluteError();
    }

    public static Supplier<TrainingMetric> maxError() {
        return BuiltInTrainingMetrics.maxError();
    }

    public static Supplier<TrainingMetric> pinballLoss(double quantile) {
        return BuiltInTrainingMetrics.pinballLoss(quantile);
    }

    public static Supplier<TrainingMetric> meanPinballLoss(double quantile) {
        return BuiltInTrainingMetrics.meanPinballLoss(quantile);
    }

    public static Supplier<TrainingMetric> predictionIntervalCoverage() {
        return BuiltInTrainingMetrics.predictionIntervalCoverage();
    }

    public static Supplier<TrainingMetric> picp() {
        return predictionIntervalCoverage();
    }

    public static Supplier<TrainingMetric> predictionIntervalMeanWidth() {
        return BuiltInTrainingMetrics.predictionIntervalMeanWidth();
    }

    public static Supplier<TrainingMetric> predictionIntervalNormalizedMeanWidth() {
        return BuiltInTrainingMetrics.predictionIntervalNormalizedMeanWidth();
    }

    public static Supplier<TrainingMetric> r2Score() {
        return BuiltInTrainingMetrics.r2Score();
    }

    public static Supplier<TrainingMetric> r2() {
        return r2Score();
    }

    public static Supplier<TrainingMetric> meanAbsolutePercentageError() {
        return BuiltInTrainingMetrics.meanAbsolutePercentageError();
    }

    public static Supplier<TrainingMetric> mape() {
        return meanAbsolutePercentageError();
    }

    public static Supplier<TrainingMetric> symmetricMeanAbsolutePercentageError() {
        return BuiltInTrainingMetrics.symmetricMeanAbsolutePercentageError();
    }

    public static Supplier<TrainingMetric> smape() {
        return symmetricMeanAbsolutePercentageError();
    }

    public static Supplier<TrainingMetric> meanBiasError() {
        return BuiltInTrainingMetrics.meanBiasError();
    }

    public static Supplier<TrainingMetric> mbe() {
        return meanBiasError();
    }

    public static Supplier<TrainingMetric> explainedVariance() {
        return BuiltInTrainingMetrics.explainedVariance();
    }

    public static Supplier<TrainingMetric> explainedVarianceScore() {
        return explainedVariance();
    }
}
