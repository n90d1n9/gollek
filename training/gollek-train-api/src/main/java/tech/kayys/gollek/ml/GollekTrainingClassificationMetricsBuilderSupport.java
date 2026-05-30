package tech.kayys.gollek.ml;

/**
 * Fluent multi-class classification metric registration for training option builders.
 */
public abstract class GollekTrainingClassificationMetricsBuilderSupport<
        B extends GollekTrainingClassificationMetricsBuilderSupport<B>>
        extends GollekTrainingMetricRegistrySupport<B> {

    public B accuracyMetric() {
        return metric(Gollek.DL.accuracyMetric());
    }

    public B confusionMatrixMetric() {
        return metric(Gollek.DL.confusionMatrixMetric());
    }

    public B classificationConfusionMatrixMetric() {
        return metric(Gollek.DL.classificationConfusionMatrixMetric());
    }

    public B topKAccuracyMetric(int k) {
        return metric(Gollek.DL.topKAccuracyMetric(k));
    }

    public B classificationLogLossMetric() {
        return metric(Gollek.DL.classificationLogLossMetric());
    }

    public B classificationCrossEntropyMetric() {
        return metric(Gollek.DL.classificationCrossEntropyMetric());
    }

    public B classificationBalancedAccuracyMetric() {
        return metric(Gollek.DL.classificationBalancedAccuracyMetric());
    }

    public B balancedAccuracyMetric() {
        return metric(Gollek.DL.balancedAccuracyMetric());
    }

    public B classificationMatthewsCorrelationCoefficientMetric() {
        return metric(Gollek.DL.classificationMatthewsCorrelationCoefficientMetric());
    }

    public B classificationMccMetric() {
        return metric(Gollek.DL.classificationMccMetric());
    }

    public B matthewsCorrelationCoefficientMetric() {
        return metric(Gollek.DL.matthewsCorrelationCoefficientMetric());
    }

    public B mccMetric() {
        return metric(Gollek.DL.mccMetric());
    }

    public B classificationWeightedPrecisionMetric() {
        return metric(Gollek.DL.classificationWeightedPrecisionMetric());
    }

    public B classificationWeightedRecallMetric() {
        return metric(Gollek.DL.classificationWeightedRecallMetric());
    }

    public B classificationWeightedF1Metric() {
        return metric(Gollek.DL.classificationWeightedF1Metric());
    }

    public B classificationCohensKappaMetric() {
        return metric(Gollek.DL.classificationCohensKappaMetric());
    }

    public B classificationKappaMetric() {
        return metric(Gollek.DL.classificationKappaMetric());
    }

    public B cohensKappaMetric() {
        return metric(Gollek.DL.cohensKappaMetric());
    }

    public B kappaMetric() {
        return metric(Gollek.DL.kappaMetric());
    }

    public B precisionMetric() {
        return metric(Gollek.DL.precisionMetric());
    }

    public B recallMetric() {
        return metric(Gollek.DL.recallMetric());
    }

    public B f1Metric() {
        return metric(Gollek.DL.f1Metric());
    }

    public B macroF1Metric() {
        return metric(Gollek.DL.macroF1Metric());
    }

    public B classificationMetrics() {
        return accuracyMetric()
                .classificationLogLossMetric()
                .precisionMetric()
                .recallMetric()
                .f1Metric();
    }

    public B classificationMacroRocAucMetric() {
        return metric(Gollek.DL.classificationMacroRocAucMetric());
    }

    public B classificationMacroAurocMetric() {
        return metric(Gollek.DL.classificationMacroAurocMetric());
    }

    public B classificationMacroAveragePrecisionMetric() {
        return metric(Gollek.DL.classificationMacroAveragePrecisionMetric());
    }

    public B classificationRankingMetrics() {
        return classificationMacroRocAucMetric()
                .classificationMacroAveragePrecisionMetric();
    }

    public B classificationImbalanceMetrics() {
        return classificationBalancedAccuracyMetric()
                .classificationMccMetric()
                .classificationWeightedPrecisionMetric()
                .classificationWeightedRecallMetric()
                .classificationWeightedF1Metric();
    }

    public B classificationAgreementMetrics() {
        return classificationKappaMetric();
    }

    public B classificationBrierScoreMetric() {
        return metric(Gollek.DL.classificationBrierScoreMetric());
    }

    public B classificationExpectedCalibrationErrorMetric() {
        return metric(Gollek.DL.classificationExpectedCalibrationErrorMetric());
    }

    public B classificationExpectedCalibrationErrorMetric(int bins) {
        return metric(Gollek.DL.classificationExpectedCalibrationErrorMetric(bins));
    }

    public B classificationCalibrationMetrics() {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric();
    }

    public B classificationCalibrationMetrics(int bins) {
        return classificationBrierScoreMetric()
                .classificationExpectedCalibrationErrorMetric(bins);
    }
}
