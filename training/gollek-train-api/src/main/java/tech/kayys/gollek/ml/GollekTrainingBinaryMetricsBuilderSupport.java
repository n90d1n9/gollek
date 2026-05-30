package tech.kayys.gollek.ml;

/**
 * Fluent binary classification metric registration for training option builders.
 */
public abstract class GollekTrainingBinaryMetricsBuilderSupport<
        B extends GollekTrainingBinaryMetricsBuilderSupport<B>>
        extends GollekTrainingClassificationMetricsBuilderSupport<B> {

    public B binaryAccuracyMetric() {
        return metric(Gollek.DL.binaryAccuracyMetric());
    }

    public B binaryAccuracyMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryAccuracyMetric(logitThreshold));
    }

    public B binaryBalancedAccuracyMetric() {
        return metric(Gollek.DL.binaryBalancedAccuracyMetric());
    }

    public B binaryBalancedAccuracyMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryBalancedAccuracyMetric(logitThreshold));
    }

    public B binaryMatthewsCorrelationCoefficientMetric() {
        return metric(Gollek.DL.binaryMatthewsCorrelationCoefficientMetric());
    }

    public B binaryMatthewsCorrelationCoefficientMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryMatthewsCorrelationCoefficientMetric(logitThreshold));
    }

    public B binaryMccMetric() {
        return metric(Gollek.DL.binaryMccMetric());
    }

    public B binaryMccMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryMccMetric(logitThreshold));
    }

    public B binaryCohensKappaMetric() {
        return metric(Gollek.DL.binaryCohensKappaMetric());
    }

    public B binaryCohensKappaMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryCohensKappaMetric(logitThreshold));
    }

    public B binaryKappaMetric() {
        return metric(Gollek.DL.binaryKappaMetric());
    }

    public B binaryKappaMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryKappaMetric(logitThreshold));
    }

    public B binaryConfusionMatrixMetric() {
        return metric(Gollek.DL.binaryConfusionMatrixMetric());
    }

    public B binaryConfusionMatrixMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryConfusionMatrixMetric(logitThreshold));
    }

    public B binaryPrecisionMetric() {
        return metric(Gollek.DL.binaryPrecisionMetric());
    }

    public B binaryPrecisionMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryPrecisionMetric(logitThreshold));
    }

    public B binaryRecallMetric() {
        return metric(Gollek.DL.binaryRecallMetric());
    }

    public B binaryRecallMetric(float logitThreshold) {
        return metric(Gollek.DL.binaryRecallMetric(logitThreshold));
    }

    public B binaryF1Metric() {
        return metric(Gollek.DL.binaryF1Metric());
    }

    public B binaryF1Metric(float logitThreshold) {
        return metric(Gollek.DL.binaryF1Metric(logitThreshold));
    }

    public B binaryRocAucMetric() {
        return metric(Gollek.DL.binaryRocAucMetric());
    }

    public B binaryAurocMetric() {
        return metric(Gollek.DL.binaryAurocMetric());
    }

    public B binaryAveragePrecisionMetric() {
        return metric(Gollek.DL.binaryAveragePrecisionMetric());
    }

    public B binaryBestF1Metric() {
        return metric(Gollek.DL.binaryBestF1Metric());
    }

    public B binaryBestF1ThresholdMetric() {
        return metric(Gollek.DL.binaryBestF1ThresholdMetric());
    }

    public B binaryPrecisionAtRecallMetric(double minimumRecall) {
        return metric(Gollek.DL.binaryPrecisionAtRecallMetric(minimumRecall));
    }

    public B binaryRecallAtPrecisionMetric(double minimumPrecision) {
        return metric(Gollek.DL.binaryRecallAtPrecisionMetric(minimumPrecision));
    }

    public B binaryBrierScoreMetric() {
        return metric(Gollek.DL.binaryBrierScoreMetric());
    }

    public B binaryLogLossMetric() {
        return metric(Gollek.DL.binaryLogLossMetric());
    }

    public B binaryCrossEntropyMetric() {
        return metric(Gollek.DL.binaryCrossEntropyMetric());
    }

    public B binaryExpectedCalibrationErrorMetric() {
        return metric(Gollek.DL.binaryExpectedCalibrationErrorMetric());
    }

    public B binaryExpectedCalibrationErrorMetric(int bins) {
        return metric(Gollek.DL.binaryExpectedCalibrationErrorMetric(bins));
    }

    public B binaryRankingMetrics() {
        return binaryRocAucMetric()
                .binaryAveragePrecisionMetric();
    }

    public B binaryImbalanceMetrics() {
        return binaryBalancedAccuracyMetric()
                .binaryMccMetric();
    }

    public B binaryImbalanceMetrics(float logitThreshold) {
        return binaryBalancedAccuracyMetric(logitThreshold)
                .binaryMccMetric(logitThreshold);
    }

    public B binaryAgreementMetrics() {
        return binaryKappaMetric();
    }

    public B binaryAgreementMetrics(float logitThreshold) {
        return binaryKappaMetric(logitThreshold);
    }

    public B binaryThresholdTuningMetrics() {
        return binaryBestF1ThresholdMetric();
    }

    public B binaryThresholdTuningMetrics(double minimumRecall, double minimumPrecision) {
        return binaryBestF1ThresholdMetric()
                .binaryPrecisionAtRecallMetric(minimumRecall)
                .binaryRecallAtPrecisionMetric(minimumPrecision);
    }

    public B binaryCalibrationMetrics() {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric();
    }

    public B binaryCalibrationMetrics(int bins) {
        return binaryBrierScoreMetric()
                .binaryLogLossMetric()
                .binaryExpectedCalibrationErrorMetric(bins);
    }

    public B binaryClassificationMetrics() {
        return binaryAccuracyMetric()
                .binaryPrecisionMetric()
                .binaryRecallMetric()
                .binaryF1Metric();
    }

    public B binaryClassificationMetrics(float logitThreshold) {
        return binaryAccuracyMetric(logitThreshold)
                .binaryPrecisionMetric(logitThreshold)
                .binaryRecallMetric(logitThreshold)
                .binaryF1Metric(logitThreshold);
    }

    public B binaryMetrics() {
        return binaryClassificationMetrics();
    }

    public B binaryMetrics(float logitThreshold) {
        return binaryClassificationMetrics(logitThreshold);
    }
}
