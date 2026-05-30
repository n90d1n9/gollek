package tech.kayys.gollek.ml;

/**
 * Fluent multi-label metric registration for training option builders.
 */
public abstract class GollekTrainingMultiLabelMetricsBuilderSupport<
        B extends GollekTrainingMultiLabelMetricsBuilderSupport<B>>
        extends GollekTrainingBinaryMetricsBuilderSupport<B> {

    public B multiLabelExactMatchMetric() {
        return metric(Gollek.DL.multiLabelExactMatchMetric());
    }

    public B multiLabelExactMatchMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelExactMatchMetric(logitThreshold));
    }

    public B multiLabelHammingLossMetric() {
        return metric(Gollek.DL.multiLabelHammingLossMetric());
    }

    public B multiLabelHammingLossMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelHammingLossMetric(logitThreshold));
    }

    public B multiLabelConfusionMatrixMetric() {
        return metric(Gollek.DL.multiLabelConfusionMatrixMetric());
    }

    public B multiLabelConfusionMatrixMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelConfusionMatrixMetric(logitThreshold));
    }

    public B multiLabelMicroPrecisionMetric() {
        return metric(Gollek.DL.multiLabelMicroPrecisionMetric());
    }

    public B multiLabelMicroPrecisionMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMicroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMicroRecallMetric() {
        return metric(Gollek.DL.multiLabelMicroRecallMetric());
    }

    public B multiLabelMicroRecallMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMicroRecallMetric(logitThreshold));
    }

    public B multiLabelMicroF1Metric() {
        return metric(Gollek.DL.multiLabelMicroF1Metric());
    }

    public B multiLabelMicroF1Metric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMicroF1Metric(logitThreshold));
    }

    public B multiLabelSamplePrecisionMetric() {
        return metric(Gollek.DL.multiLabelSamplePrecisionMetric());
    }

    public B multiLabelSamplePrecisionMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelSamplePrecisionMetric(logitThreshold));
    }

    public B multiLabelSampleRecallMetric() {
        return metric(Gollek.DL.multiLabelSampleRecallMetric());
    }

    public B multiLabelSampleRecallMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelSampleRecallMetric(logitThreshold));
    }

    public B multiLabelSampleF1Metric() {
        return metric(Gollek.DL.multiLabelSampleF1Metric());
    }

    public B multiLabelSampleF1Metric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelSampleF1Metric(logitThreshold));
    }

    public B multiLabelSampleJaccardMetric() {
        return metric(Gollek.DL.multiLabelSampleJaccardMetric());
    }

    public B multiLabelSampleJaccardMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelSampleJaccardMetric(logitThreshold));
    }

    public B multiLabelMacroPrecisionMetric() {
        return metric(Gollek.DL.multiLabelMacroPrecisionMetric());
    }

    public B multiLabelMacroPrecisionMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMacroPrecisionMetric(logitThreshold));
    }

    public B multiLabelMacroRecallMetric() {
        return metric(Gollek.DL.multiLabelMacroRecallMetric());
    }

    public B multiLabelMacroRecallMetric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMacroRecallMetric(logitThreshold));
    }

    public B multiLabelMacroF1Metric() {
        return metric(Gollek.DL.multiLabelMacroF1Metric());
    }

    public B multiLabelMacroF1Metric(float logitThreshold) {
        return metric(Gollek.DL.multiLabelMacroF1Metric(logitThreshold));
    }

    public B multiLabelMacroRocAucMetric() {
        return metric(Gollek.DL.multiLabelMacroRocAucMetric());
    }

    public B multiLabelMacroAurocMetric() {
        return metric(Gollek.DL.multiLabelMacroAurocMetric());
    }

    public B multiLabelMacroAveragePrecisionMetric() {
        return metric(Gollek.DL.multiLabelMacroAveragePrecisionMetric());
    }

    public B multiLabelLabelRankingAveragePrecisionMetric() {
        return metric(Gollek.DL.multiLabelLabelRankingAveragePrecisionMetric());
    }

    public B multiLabelLrapMetric() {
        return metric(Gollek.DL.multiLabelLrapMetric());
    }

    public B multiLabelRankingLossMetric() {
        return metric(Gollek.DL.multiLabelRankingLossMetric());
    }

    public B multiLabelCoverageErrorMetric() {
        return metric(Gollek.DL.multiLabelCoverageErrorMetric());
    }

    public B multiLabelMacroBestF1Metric() {
        return metric(Gollek.DL.multiLabelMacroBestF1Metric());
    }

    public B multiLabelBestF1ThresholdsMetric() {
        return metric(Gollek.DL.multiLabelBestF1ThresholdsMetric());
    }

    public B multiLabelRankingMetrics() {
        return multiLabelMacroRocAucMetric()
                .multiLabelMacroAveragePrecisionMetric()
                .multiLabelLabelRankingAveragePrecisionMetric()
                .multiLabelRankingLossMetric()
                .multiLabelCoverageErrorMetric();
    }

    public B multiLabelThresholdTuningMetrics() {
        return multiLabelBestF1ThresholdsMetric();
    }

    public B multiLabelMicroMetrics() {
        return multiLabelMicroPrecisionMetric()
                .multiLabelMicroRecallMetric()
                .multiLabelMicroF1Metric();
    }

    public B multiLabelMicroMetrics(float logitThreshold) {
        return multiLabelMicroPrecisionMetric(logitThreshold)
                .multiLabelMicroRecallMetric(logitThreshold)
                .multiLabelMicroF1Metric(logitThreshold);
    }

    public B multiLabelSampleMetrics() {
        return multiLabelSamplePrecisionMetric()
                .multiLabelSampleRecallMetric()
                .multiLabelSampleF1Metric()
                .multiLabelSampleJaccardMetric();
    }

    public B multiLabelSampleMetrics(float logitThreshold) {
        return multiLabelSamplePrecisionMetric(logitThreshold)
                .multiLabelSampleRecallMetric(logitThreshold)
                .multiLabelSampleF1Metric(logitThreshold)
                .multiLabelSampleJaccardMetric(logitThreshold);
    }

    public B multiLabelBinaryMetrics() {
        return multiLabelExactMatchMetric()
                .multiLabelHammingLossMetric()
                .multiLabelMicroMetrics()
                .multiLabelSampleMetrics()
                .multiLabelMacroPrecisionMetric()
                .multiLabelMacroRecallMetric()
                .multiLabelMacroF1Metric();
    }

    public B multiLabelBinaryMetrics(float logitThreshold) {
        return multiLabelExactMatchMetric(logitThreshold)
                .multiLabelHammingLossMetric(logitThreshold)
                .multiLabelMicroMetrics(logitThreshold)
                .multiLabelSampleMetrics(logitThreshold)
                .multiLabelMacroPrecisionMetric(logitThreshold)
                .multiLabelMacroRecallMetric(logitThreshold)
                .multiLabelMacroF1Metric(logitThreshold);
    }

    public B multiLabelMetrics() {
        return multiLabelBinaryMetrics();
    }

    public B multiLabelMetrics(float logitThreshold) {
        return multiLabelBinaryMetrics(logitThreshold);
    }
}
