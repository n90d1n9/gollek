package tech.kayys.gollek.ml;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Public binary classification metric factories inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlBinaryMetricsFacade extends GollekDlClassificationMetricsFacade {
    protected GollekDlBinaryMetricsFacade() {
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric() {
        return CanonicalTrainer.Metrics.binaryAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAccuracyMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryAccuracy(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBalancedAccuracyMetric() {
        return CanonicalTrainer.Metrics.binaryBalancedAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBalancedAccuracyMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryBalancedAccuracy(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMatthewsCorrelationCoefficientMetric() {
        return CanonicalTrainer.Metrics.binaryMatthewsCorrelationCoefficient();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMatthewsCorrelationCoefficientMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryMatthewsCorrelationCoefficient(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMccMetric() {
        return CanonicalTrainer.Metrics.binaryMcc();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryMccMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryMcc(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCohensKappaMetric() {
        return CanonicalTrainer.Metrics.binaryCohensKappa();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCohensKappaMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryCohensKappa(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryKappaMetric() {
        return CanonicalTrainer.Metrics.binaryKappa();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryKappaMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryKappa(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric() {
        return CanonicalTrainer.Metrics.binaryConfusionMatrix();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryConfusionMatrixMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryConfusionMatrix(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric() {
        return CanonicalTrainer.Metrics.binaryPrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecisionMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryPrecision(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecallMetric() {
        return CanonicalTrainer.Metrics.binaryRecall();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecallMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryRecall(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1Metric() {
        return CanonicalTrainer.Metrics.binaryF1();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryF1Metric(float logitThreshold) {
        return CanonicalTrainer.Metrics.binaryF1(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRocAucMetric() {
        return CanonicalTrainer.Metrics.binaryRocAuc();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAurocMetric() {
        return CanonicalTrainer.Metrics.binaryAuroc();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryAveragePrecisionMetric() {
        return CanonicalTrainer.Metrics.binaryAveragePrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBestF1Metric() {
        return CanonicalTrainer.Metrics.binaryBestF1();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBestF1ThresholdMetric() {
        return CanonicalTrainer.Metrics.binaryBestF1Threshold();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryPrecisionAtRecallMetric(double minimumRecall) {
        return CanonicalTrainer.Metrics.binaryPrecisionAtRecall(minimumRecall);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryRecallAtPrecisionMetric(double minimumPrecision) {
        return CanonicalTrainer.Metrics.binaryRecallAtPrecision(minimumPrecision);
    }

    public static Supplier<CanonicalTrainer.Metric> binaryBrierScoreMetric() {
        return CanonicalTrainer.Metrics.binaryBrierScore();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryLogLossMetric() {
        return CanonicalTrainer.Metrics.binaryLogLoss();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryCrossEntropyMetric() {
        return CanonicalTrainer.Metrics.binaryCrossEntropy();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryExpectedCalibrationErrorMetric() {
        return CanonicalTrainer.Metrics.binaryExpectedCalibrationError();
    }

    public static Supplier<CanonicalTrainer.Metric> binaryExpectedCalibrationErrorMetric(int bins) {
        return CanonicalTrainer.Metrics.binaryExpectedCalibrationError(bins);
    }
}
