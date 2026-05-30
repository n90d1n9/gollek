package tech.kayys.gollek.ml.train;

import java.util.function.Supplier;

/**
 * Package-private facade behind the public {@link TrainingMetrics} catalog.
 */
final class BuiltInTrainingMetrics {
    private BuiltInTrainingMetrics() {
    }

    static Supplier<TrainingMetric> classificationAccuracy() {
        return ClassificationTrainingMetrics.classificationAccuracy();
    }

    static Supplier<TrainingMetric> accuracy() {
        return ClassificationTrainingMetrics.accuracy();
    }

    static Supplier<TrainingMetric> classificationConfusionMatrix() {
        return ClassificationTrainingMetrics.classificationConfusionMatrix();
    }

    static Supplier<TrainingMetric> confusionMatrix() {
        return ClassificationTrainingMetrics.confusionMatrix();
    }

    static Supplier<TrainingMetric> topKAccuracy(int k) {
        return ClassificationTrainingMetrics.topKAccuracy(k);
    }

    static Supplier<TrainingMetric> classificationLogLoss() {
        return ClassificationTrainingMetrics.classificationLogLoss();
    }

    static Supplier<TrainingMetric> classificationCrossEntropy() {
        return ClassificationTrainingMetrics.classificationCrossEntropy();
    }

    static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy() {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingTokenAccuracy();
    }

    static Supplier<TrainingMetric> causalLanguageModelingTokenAccuracy(float ignoreIndex) {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingTokenAccuracy(ignoreIndex);
    }

    static Supplier<TrainingMetric> causalLanguageModelingLogLoss() {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingLogLoss();
    }

    static Supplier<TrainingMetric> causalLanguageModelingLogLoss(float ignoreIndex) {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingLogLoss(ignoreIndex);
    }

    static Supplier<TrainingMetric> causalLanguageModelingPerplexity() {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingPerplexity();
    }

    static Supplier<TrainingMetric> causalLanguageModelingPerplexity(float ignoreIndex) {
        return CausalLanguageModelingTrainingMetrics.causalLanguageModelingPerplexity(ignoreIndex);
    }

    static Supplier<TrainingMetric> classificationBalancedAccuracy() {
        return ClassificationImbalanceTrainingMetrics.classificationBalancedAccuracy();
    }

    static Supplier<TrainingMetric> balancedAccuracy() {
        return classificationBalancedAccuracy();
    }

    static Supplier<TrainingMetric> classificationMatthewsCorrelationCoefficient() {
        return ClassificationImbalanceTrainingMetrics.classificationMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> classificationMcc() {
        return classificationMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> matthewsCorrelationCoefficient() {
        return classificationMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> mcc() {
        return classificationMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> classificationWeightedPrecision() {
        return ClassificationImbalanceTrainingMetrics.classificationWeightedPrecision();
    }

    static Supplier<TrainingMetric> classificationWeightedRecall() {
        return ClassificationImbalanceTrainingMetrics.classificationWeightedRecall();
    }

    static Supplier<TrainingMetric> classificationWeightedF1() {
        return ClassificationImbalanceTrainingMetrics.classificationWeightedF1();
    }

    static Supplier<TrainingMetric> classificationCohensKappa() {
        return ClassificationImbalanceTrainingMetrics.classificationCohensKappa();
    }

    static Supplier<TrainingMetric> classificationKappa() {
        return ClassificationImbalanceTrainingMetrics.classificationKappa();
    }

    static Supplier<TrainingMetric> cohensKappa() {
        return classificationCohensKappa();
    }

    static Supplier<TrainingMetric> kappa() {
        return classificationCohensKappa();
    }

    static Supplier<TrainingMetric> classificationBrierScore() {
        return ClassificationCalibrationTrainingMetrics.classificationBrierScore();
    }

    static Supplier<TrainingMetric> classificationExpectedCalibrationError() {
        return ClassificationCalibrationTrainingMetrics.classificationExpectedCalibrationError();
    }

    static Supplier<TrainingMetric> classificationExpectedCalibrationError(int bins) {
        return ClassificationCalibrationTrainingMetrics.classificationExpectedCalibrationError(bins);
    }

    static Supplier<TrainingMetric> binaryAccuracy() {
        return BinaryTrainingMetrics.binaryAccuracy();
    }

    static Supplier<TrainingMetric> binaryAccuracy(float logitThreshold) {
        return BinaryTrainingMetrics.binaryAccuracy(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryBalancedAccuracy() {
        return BinaryTrainingMetrics.binaryBalancedAccuracy();
    }

    static Supplier<TrainingMetric> binaryBalancedAccuracy(float logitThreshold) {
        return BinaryTrainingMetrics.binaryBalancedAccuracy(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient() {
        return BinaryTrainingMetrics.binaryMatthewsCorrelationCoefficient();
    }

    static Supplier<TrainingMetric> binaryMatthewsCorrelationCoefficient(float logitThreshold) {
        return BinaryTrainingMetrics.binaryMatthewsCorrelationCoefficient(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryMcc() {
        return BinaryTrainingMetrics.binaryMcc();
    }

    static Supplier<TrainingMetric> binaryMcc(float logitThreshold) {
        return BinaryTrainingMetrics.binaryMcc(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryCohensKappa() {
        return BinaryTrainingMetrics.binaryCohensKappa();
    }

    static Supplier<TrainingMetric> binaryCohensKappa(float logitThreshold) {
        return BinaryTrainingMetrics.binaryCohensKappa(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryKappa() {
        return BinaryTrainingMetrics.binaryKappa();
    }

    static Supplier<TrainingMetric> binaryKappa(float logitThreshold) {
        return BinaryTrainingMetrics.binaryKappa(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix() {
        return BinaryTrainingMetrics.binaryConfusionMatrix();
    }

    static Supplier<TrainingMetric> binaryConfusionMatrix(float logitThreshold) {
        return BinaryTrainingMetrics.binaryConfusionMatrix(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryPrecision() {
        return BinaryTrainingMetrics.binaryPrecision();
    }

    static Supplier<TrainingMetric> binaryPrecision(float logitThreshold) {
        return BinaryTrainingMetrics.binaryPrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryRecall() {
        return BinaryTrainingMetrics.binaryRecall();
    }

    static Supplier<TrainingMetric> binaryRecall(float logitThreshold) {
        return BinaryTrainingMetrics.binaryRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryF1() {
        return BinaryTrainingMetrics.binaryF1();
    }

    static Supplier<TrainingMetric> binaryF1(float logitThreshold) {
        return BinaryTrainingMetrics.binaryF1(logitThreshold);
    }

    static Supplier<TrainingMetric> binaryRocAuc() {
        return BinaryTrainingMetrics.binaryRocAuc();
    }

    static Supplier<TrainingMetric> binaryAuroc() {
        return BinaryTrainingMetrics.binaryAuroc();
    }

    static Supplier<TrainingMetric> binaryAveragePrecision() {
        return BinaryTrainingMetrics.binaryAveragePrecision();
    }

    static Supplier<TrainingMetric> binaryBestF1() {
        return BinaryTrainingMetrics.binaryBestF1();
    }

    static Supplier<TrainingMetric> binaryBestF1Threshold() {
        return BinaryTrainingMetrics.binaryBestF1Threshold();
    }

    static Supplier<TrainingMetric> binaryPrecisionAtRecall(double minimumRecall) {
        return BinaryTrainingMetrics.binaryPrecisionAtRecall(minimumRecall);
    }

    static Supplier<TrainingMetric> binaryRecallAtPrecision(double minimumPrecision) {
        return BinaryTrainingMetrics.binaryRecallAtPrecision(minimumPrecision);
    }

    static Supplier<TrainingMetric> binaryBrierScore() {
        return BinaryCalibrationTrainingMetrics.binaryBrierScore();
    }

    static Supplier<TrainingMetric> binaryLogLoss() {
        return BinaryCalibrationTrainingMetrics.binaryLogLoss();
    }

    static Supplier<TrainingMetric> binaryCrossEntropy() {
        return BinaryCalibrationTrainingMetrics.binaryCrossEntropy();
    }

    static Supplier<TrainingMetric> binaryExpectedCalibrationError() {
        return BinaryCalibrationTrainingMetrics.binaryExpectedCalibrationError();
    }

    static Supplier<TrainingMetric> binaryExpectedCalibrationError(int bins) {
        return BinaryCalibrationTrainingMetrics.binaryExpectedCalibrationError(bins);
    }

    static Supplier<TrainingMetric> multiLabelExactMatch() {
        return MultiLabelTrainingMetrics.multiLabelExactMatch();
    }

    static Supplier<TrainingMetric> multiLabelExactMatch(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelExactMatch(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss() {
        return MultiLabelTrainingMetrics.multiLabelHammingLoss();
    }

    static Supplier<TrainingMetric> multiLabelHammingLoss(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelHammingLoss(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelConfusionMatrix() {
        return MultiLabelTrainingMetrics.multiLabelConfusionMatrix();
    }

    static Supplier<TrainingMetric> multiLabelConfusionMatrix(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelConfusionMatrix(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroPrecision() {
        return MultiLabelTrainingMetrics.multiLabelMicroPrecision();
    }

    static Supplier<TrainingMetric> multiLabelMicroPrecision(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMicroPrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroRecall() {
        return MultiLabelTrainingMetrics.multiLabelMicroRecall();
    }

    static Supplier<TrainingMetric> multiLabelMicroRecall(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMicroRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMicroF1() {
        return MultiLabelTrainingMetrics.multiLabelMicroF1();
    }

    static Supplier<TrainingMetric> multiLabelMicroF1(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMicroF1(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelSamplePrecision() {
        return MultiLabelTrainingMetrics.multiLabelSamplePrecision();
    }

    static Supplier<TrainingMetric> multiLabelSamplePrecision(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelSamplePrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleRecall() {
        return MultiLabelTrainingMetrics.multiLabelSampleRecall();
    }

    static Supplier<TrainingMetric> multiLabelSampleRecall(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelSampleRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleF1() {
        return MultiLabelTrainingMetrics.multiLabelSampleF1();
    }

    static Supplier<TrainingMetric> multiLabelSampleF1(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelSampleF1(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelSampleJaccard() {
        return MultiLabelTrainingMetrics.multiLabelSampleJaccard();
    }

    static Supplier<TrainingMetric> multiLabelSampleJaccard(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelSampleJaccard(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision() {
        return MultiLabelTrainingMetrics.multiLabelMacroPrecision();
    }

    static Supplier<TrainingMetric> multiLabelMacroPrecision(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroPrecision(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall() {
        return MultiLabelTrainingMetrics.multiLabelMacroRecall();
    }

    static Supplier<TrainingMetric> multiLabelMacroRecall(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroRecall(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroF1() {
        return MultiLabelTrainingMetrics.multiLabelMacroF1();
    }

    static Supplier<TrainingMetric> multiLabelMacroF1(float logitThreshold) {
        return MultiLabelTrainingMetrics.multiLabelMacroF1(logitThreshold);
    }

    static Supplier<TrainingMetric> multiLabelMacroRocAuc() {
        return MultiLabelTrainingMetrics.multiLabelMacroRocAuc();
    }

    static Supplier<TrainingMetric> multiLabelMacroAuroc() {
        return MultiLabelTrainingMetrics.multiLabelMacroAuroc();
    }

    static Supplier<TrainingMetric> multiLabelMacroAveragePrecision() {
        return MultiLabelTrainingMetrics.multiLabelMacroAveragePrecision();
    }

    static Supplier<TrainingMetric> multiLabelLabelRankingAveragePrecision() {
        return MultiLabelRankingTrainingMetrics.multiLabelLabelRankingAveragePrecision();
    }

    static Supplier<TrainingMetric> multiLabelLrap() {
        return multiLabelLabelRankingAveragePrecision();
    }

    static Supplier<TrainingMetric> multiLabelRankingLoss() {
        return MultiLabelRankingTrainingMetrics.multiLabelRankingLoss();
    }

    static Supplier<TrainingMetric> multiLabelCoverageError() {
        return MultiLabelRankingTrainingMetrics.multiLabelCoverageError();
    }

    static Supplier<TrainingMetric> multiLabelMacroBestF1() {
        return MultiLabelThresholdTrainingMetrics.multiLabelMacroBestF1();
    }

    static Supplier<TrainingMetric> multiLabelBestF1Thresholds() {
        return MultiLabelThresholdTrainingMetrics.multiLabelBestF1Thresholds();
    }

    static Supplier<TrainingMetric> precision() {
        return ClassificationTrainingMetrics.precision();
    }

    static Supplier<TrainingMetric> recall() {
        return ClassificationTrainingMetrics.recall();
    }

    static Supplier<TrainingMetric> f1() {
        return ClassificationTrainingMetrics.f1();
    }

    static Supplier<TrainingMetric> macroF1() {
        return ClassificationTrainingMetrics.macroF1();
    }

    static Supplier<TrainingMetric> classificationMacroRocAuc() {
        return ClassificationTrainingMetrics.classificationMacroRocAuc();
    }

    static Supplier<TrainingMetric> classificationMacroAuroc() {
        return ClassificationTrainingMetrics.classificationMacroAuroc();
    }

    static Supplier<TrainingMetric> classificationMacroAveragePrecision() {
        return ClassificationTrainingMetrics.classificationMacroAveragePrecision();
    }

    static Supplier<TrainingMetric> meanAbsoluteError() {
        return RegressionTrainingMetrics.meanAbsoluteError();
    }

    static Supplier<TrainingMetric> mae() {
        return RegressionTrainingMetrics.mae();
    }

    static Supplier<TrainingMetric> meanSquaredError() {
        return RegressionTrainingMetrics.meanSquaredError();
    }

    static Supplier<TrainingMetric> mse() {
        return RegressionTrainingMetrics.mse();
    }

    static Supplier<TrainingMetric> rootMeanSquaredError() {
        return RegressionTrainingMetrics.rootMeanSquaredError();
    }

    static Supplier<TrainingMetric> rmse() {
        return RegressionTrainingMetrics.rmse();
    }

    static Supplier<TrainingMetric> meanSquaredLogError() {
        return RegressionTrainingMetrics.meanSquaredLogError();
    }

    static Supplier<TrainingMetric> msle() {
        return RegressionTrainingMetrics.msle();
    }

    static Supplier<TrainingMetric> rootMeanSquaredLogError() {
        return RegressionTrainingMetrics.rootMeanSquaredLogError();
    }

    static Supplier<TrainingMetric> rmsle() {
        return RegressionTrainingMetrics.rmsle();
    }

    static Supplier<TrainingMetric> meanPoissonDeviance() {
        return CountRegressionTrainingMetrics.meanPoissonDeviance();
    }

    static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput) {
        return CountRegressionTrainingMetrics.meanPoissonDeviance(logInput);
    }

    static Supplier<TrainingMetric> meanPoissonDeviance(boolean logInput, double eps) {
        return CountRegressionTrainingMetrics.meanPoissonDeviance(logInput, eps);
    }

    static Supplier<TrainingMetric> poissonDeviance() {
        return CountRegressionTrainingMetrics.poissonDeviance();
    }

    static Supplier<TrainingMetric> poissonLogRateDeviance() {
        return CountRegressionTrainingMetrics.poissonLogRateDeviance();
    }

    static Supplier<TrainingMetric> meanTweedieDeviance() {
        return CountRegressionTrainingMetrics.meanTweedieDeviance();
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power) {
        return CountRegressionTrainingMetrics.meanTweedieDeviance(power);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput) {
        return CountRegressionTrainingMetrics.meanTweedieDeviance(power, logInput);
    }

    static Supplier<TrainingMetric> meanTweedieDeviance(double power, boolean logInput, double eps) {
        return CountRegressionTrainingMetrics.meanTweedieDeviance(power, logInput, eps);
    }

    static Supplier<TrainingMetric> tweedieDeviance() {
        return CountRegressionTrainingMetrics.tweedieDeviance();
    }

    static Supplier<TrainingMetric> compoundPoissonGammaDeviance() {
        return CountRegressionTrainingMetrics.compoundPoissonGammaDeviance();
    }

    static Supplier<TrainingMetric> medianAbsoluteError() {
        return RegressionTrainingMetrics.medianAbsoluteError();
    }

    static Supplier<TrainingMetric> medae() {
        return RegressionTrainingMetrics.medae();
    }

    static Supplier<TrainingMetric> maxError() {
        return RegressionTrainingMetrics.maxError();
    }

    static Supplier<TrainingMetric> pinballLoss(double quantile) {
        return RegressionTrainingMetrics.pinballLoss(quantile);
    }

    static Supplier<TrainingMetric> meanPinballLoss(double quantile) {
        return RegressionTrainingMetrics.meanPinballLoss(quantile);
    }

    static Supplier<TrainingMetric> predictionIntervalCoverage() {
        return RegressionTrainingMetrics.predictionIntervalCoverage();
    }

    static Supplier<TrainingMetric> picp() {
        return RegressionTrainingMetrics.picp();
    }

    static Supplier<TrainingMetric> predictionIntervalMeanWidth() {
        return RegressionTrainingMetrics.predictionIntervalMeanWidth();
    }

    static Supplier<TrainingMetric> predictionIntervalNormalizedMeanWidth() {
        return RegressionTrainingMetrics.predictionIntervalNormalizedMeanWidth();
    }

    static Supplier<TrainingMetric> r2Score() {
        return RegressionTrainingMetrics.r2Score();
    }

    static Supplier<TrainingMetric> r2() {
        return RegressionTrainingMetrics.r2();
    }

    static Supplier<TrainingMetric> meanAbsolutePercentageError() {
        return RegressionTrainingMetrics.meanAbsolutePercentageError();
    }

    static Supplier<TrainingMetric> mape() {
        return RegressionTrainingMetrics.mape();
    }

    static Supplier<TrainingMetric> symmetricMeanAbsolutePercentageError() {
        return RegressionTrainingMetrics.symmetricMeanAbsolutePercentageError();
    }

    static Supplier<TrainingMetric> smape() {
        return RegressionTrainingMetrics.smape();
    }

    static Supplier<TrainingMetric> meanBiasError() {
        return RegressionTrainingMetrics.meanBiasError();
    }

    static Supplier<TrainingMetric> mbe() {
        return RegressionTrainingMetrics.mbe();
    }

    static Supplier<TrainingMetric> explainedVariance() {
        return RegressionTrainingMetrics.explainedVariance();
    }

    static Supplier<TrainingMetric> explainedVarianceScore() {
        return RegressionTrainingMetrics.explainedVarianceScore();
    }
}
