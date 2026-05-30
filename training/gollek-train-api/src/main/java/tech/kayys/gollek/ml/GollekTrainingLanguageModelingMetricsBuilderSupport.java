package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.nn.loss.CausalLanguageModelingLoss;

/**
 * Fluent next-token language-modeling metric registration for training option builders.
 */
public abstract class GollekTrainingLanguageModelingMetricsBuilderSupport<
        B extends GollekTrainingLanguageModelingMetricsBuilderSupport<B>>
        extends GollekTrainingRegressionMetricsBuilderSupport<B> {

    public B causalLanguageModelingTokenAccuracyMetric() {
        return metric(Gollek.DL.causalLanguageModelingTokenAccuracyMetric());
    }

    public B causalLanguageModelingTokenAccuracyMetric(float ignoreIndex) {
        return metric(Gollek.DL.causalLanguageModelingTokenAccuracyMetric(ignoreIndex));
    }

    public B nextTokenAccuracyMetric() {
        return metric(Gollek.DL.nextTokenAccuracyMetric());
    }

    public B nextTokenAccuracyMetric(float ignoreIndex) {
        return metric(Gollek.DL.nextTokenAccuracyMetric(ignoreIndex));
    }

    public B causalLanguageModelingLogLossMetric() {
        return metric(Gollek.DL.causalLanguageModelingLogLossMetric());
    }

    public B causalLanguageModelingLogLossMetric(float ignoreIndex) {
        return metric(Gollek.DL.causalLanguageModelingLogLossMetric(ignoreIndex));
    }

    public B causalLanguageModelingCrossEntropyMetric() {
        return metric(Gollek.DL.causalLanguageModelingCrossEntropyMetric());
    }

    public B causalLanguageModelingCrossEntropyMetric(float ignoreIndex) {
        return metric(Gollek.DL.causalLanguageModelingCrossEntropyMetric(ignoreIndex));
    }

    public B causalLanguageModelingPerplexityMetric() {
        return metric(Gollek.DL.causalLanguageModelingPerplexityMetric());
    }

    public B causalLanguageModelingPerplexityMetric(float ignoreIndex) {
        return metric(Gollek.DL.causalLanguageModelingPerplexityMetric(ignoreIndex));
    }

    public B nextTokenPerplexityMetric() {
        return metric(Gollek.DL.nextTokenPerplexityMetric());
    }

    public B nextTokenPerplexityMetric(float ignoreIndex) {
        return metric(Gollek.DL.nextTokenPerplexityMetric(ignoreIndex));
    }

    public B causalLanguageModelingMetrics() {
        return causalLanguageModelingTokenAccuracyMetric()
                .causalLanguageModelingLogLossMetric()
                .causalLanguageModelingPerplexityMetric();
    }

    public B causalLanguageModelingMetrics(float ignoreIndex) {
        return causalLanguageModelingTokenAccuracyMetric(ignoreIndex)
                .causalLanguageModelingLogLossMetric(ignoreIndex)
                .causalLanguageModelingPerplexityMetric(ignoreIndex);
    }

    public B causalLanguageModelingDefaults() {
        float ignoreIndex = causalLanguageModelingIgnoreIndex == null
                ? CausalLanguageModelingLoss.DEFAULT_IGNORE_INDEX
                : causalLanguageModelingIgnoreIndex;
        return causalLanguageModelingMetrics(ignoreIndex);
    }

    public B causalLanguageModelingDefaults(float ignoreIndex) {
        return causalLanguageModelingIgnoreIndex(ignoreIndex)
                .causalLanguageModelingMetrics(ignoreIndex);
    }

    public B nextTokenMetrics() {
        return causalLanguageModelingMetrics();
    }

    public B nextTokenMetrics(float ignoreIndex) {
        return causalLanguageModelingMetrics(ignoreIndex);
    }

    public B nextTokenDefaults() {
        return causalLanguageModelingDefaults();
    }

    public B nextTokenDefaults(float ignoreIndex) {
        return causalLanguageModelingDefaults(ignoreIndex);
    }
}
