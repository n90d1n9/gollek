package tech.kayys.gollek.ml;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Public multi-class classification metric factories inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlClassificationMetricsFacade {
    protected GollekDlClassificationMetricsFacade() {
    }

    public static Supplier<CanonicalTrainer.Metric> accuracyMetric() {
        return CanonicalTrainer.Metrics.accuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> confusionMatrixMetric() {
        return CanonicalTrainer.Metrics.confusionMatrix();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationConfusionMatrixMetric() {
        return CanonicalTrainer.Metrics.classificationConfusionMatrix();
    }

    public static Supplier<CanonicalTrainer.Metric> topKAccuracyMetric(int k) {
        return CanonicalTrainer.Metrics.topKAccuracy(k);
    }

    public static Supplier<CanonicalTrainer.Metric> classificationLogLossMetric() {
        return CanonicalTrainer.Metrics.classificationLogLoss();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationCrossEntropyMetric() {
        return CanonicalTrainer.Metrics.classificationCrossEntropy();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationBalancedAccuracyMetric() {
        return CanonicalTrainer.Metrics.classificationBalancedAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> balancedAccuracyMetric() {
        return CanonicalTrainer.Metrics.balancedAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMatthewsCorrelationCoefficientMetric() {
        return CanonicalTrainer.Metrics.classificationMatthewsCorrelationCoefficient();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMccMetric() {
        return CanonicalTrainer.Metrics.classificationMcc();
    }

    public static Supplier<CanonicalTrainer.Metric> matthewsCorrelationCoefficientMetric() {
        return CanonicalTrainer.Metrics.matthewsCorrelationCoefficient();
    }

    public static Supplier<CanonicalTrainer.Metric> mccMetric() {
        return CanonicalTrainer.Metrics.mcc();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedPrecisionMetric() {
        return CanonicalTrainer.Metrics.classificationWeightedPrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedRecallMetric() {
        return CanonicalTrainer.Metrics.classificationWeightedRecall();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationWeightedF1Metric() {
        return CanonicalTrainer.Metrics.classificationWeightedF1();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationCohensKappaMetric() {
        return CanonicalTrainer.Metrics.classificationCohensKappa();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationKappaMetric() {
        return CanonicalTrainer.Metrics.classificationKappa();
    }

    public static Supplier<CanonicalTrainer.Metric> cohensKappaMetric() {
        return CanonicalTrainer.Metrics.cohensKappa();
    }

    public static Supplier<CanonicalTrainer.Metric> kappaMetric() {
        return CanonicalTrainer.Metrics.kappa();
    }

    public static Supplier<CanonicalTrainer.Metric> precisionMetric() {
        return CanonicalTrainer.Metrics.precision();
    }

    public static Supplier<CanonicalTrainer.Metric> recallMetric() {
        return CanonicalTrainer.Metrics.recall();
    }

    public static Supplier<CanonicalTrainer.Metric> f1Metric() {
        return CanonicalTrainer.Metrics.f1();
    }

    public static Supplier<CanonicalTrainer.Metric> macroF1Metric() {
        return CanonicalTrainer.Metrics.macroF1();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroRocAucMetric() {
        return CanonicalTrainer.Metrics.classificationMacroRocAuc();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAurocMetric() {
        return CanonicalTrainer.Metrics.classificationMacroAuroc();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationMacroAveragePrecisionMetric() {
        return CanonicalTrainer.Metrics.classificationMacroAveragePrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationBrierScoreMetric() {
        return CanonicalTrainer.Metrics.classificationBrierScore();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationExpectedCalibrationErrorMetric() {
        return CanonicalTrainer.Metrics.classificationExpectedCalibrationError();
    }

    public static Supplier<CanonicalTrainer.Metric> classificationExpectedCalibrationErrorMetric(int bins) {
        return CanonicalTrainer.Metrics.classificationExpectedCalibrationError(bins);
    }
}
