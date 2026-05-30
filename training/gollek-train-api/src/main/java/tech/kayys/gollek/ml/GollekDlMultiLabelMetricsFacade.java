package tech.kayys.gollek.ml;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Public multi-label metric factories inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlMultiLabelMetricsFacade extends GollekDlBinaryMetricsFacade {
    protected GollekDlMultiLabelMetricsFacade() {
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric() {
        return CanonicalTrainer.Metrics.multiLabelExactMatch();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelExactMatchMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelExactMatch(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric() {
        return CanonicalTrainer.Metrics.multiLabelHammingLoss();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelHammingLossMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelHammingLoss(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelConfusionMatrixMetric() {
        return CanonicalTrainer.Metrics.multiLabelConfusionMatrix();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelConfusionMatrixMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelConfusionMatrix(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroPrecisionMetric() {
        return CanonicalTrainer.Metrics.multiLabelMicroPrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroPrecisionMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMicroPrecision(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroRecallMetric() {
        return CanonicalTrainer.Metrics.multiLabelMicroRecall();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroRecallMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMicroRecall(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroF1Metric() {
        return CanonicalTrainer.Metrics.multiLabelMicroF1();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMicroF1Metric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMicroF1(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSamplePrecisionMetric() {
        return CanonicalTrainer.Metrics.multiLabelSamplePrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSamplePrecisionMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelSamplePrecision(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleRecallMetric() {
        return CanonicalTrainer.Metrics.multiLabelSampleRecall();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleRecallMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelSampleRecall(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleF1Metric() {
        return CanonicalTrainer.Metrics.multiLabelSampleF1();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleF1Metric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelSampleF1(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleJaccardMetric() {
        return CanonicalTrainer.Metrics.multiLabelSampleJaccard();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelSampleJaccardMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelSampleJaccard(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric() {
        return CanonicalTrainer.Metrics.multiLabelMacroPrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroPrecisionMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMacroPrecision(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric() {
        return CanonicalTrainer.Metrics.multiLabelMacroRecall();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRecallMetric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMacroRecall(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric() {
        return CanonicalTrainer.Metrics.multiLabelMacroF1();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroF1Metric(float logitThreshold) {
        return CanonicalTrainer.Metrics.multiLabelMacroF1(logitThreshold);
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroRocAucMetric() {
        return CanonicalTrainer.Metrics.multiLabelMacroRocAuc();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAurocMetric() {
        return CanonicalTrainer.Metrics.multiLabelMacroAuroc();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroAveragePrecisionMetric() {
        return CanonicalTrainer.Metrics.multiLabelMacroAveragePrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelLabelRankingAveragePrecisionMetric() {
        return CanonicalTrainer.Metrics.multiLabelLabelRankingAveragePrecision();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelLrapMetric() {
        return CanonicalTrainer.Metrics.multiLabelLrap();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelRankingLossMetric() {
        return CanonicalTrainer.Metrics.multiLabelRankingLoss();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelCoverageErrorMetric() {
        return CanonicalTrainer.Metrics.multiLabelCoverageError();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelMacroBestF1Metric() {
        return CanonicalTrainer.Metrics.multiLabelMacroBestF1();
    }

    public static Supplier<CanonicalTrainer.Metric> multiLabelBestF1ThresholdsMetric() {
        return CanonicalTrainer.Metrics.multiLabelBestF1Thresholds();
    }
}
