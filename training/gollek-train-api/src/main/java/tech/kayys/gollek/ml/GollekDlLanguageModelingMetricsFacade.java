package tech.kayys.gollek.ml;

import java.util.function.Supplier;
import tech.kayys.gollek.ml.train.CanonicalTrainer;

/**
 * Public next-token language-modeling metric factories inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlLanguageModelingMetricsFacade extends GollekDlRegressionMetricsFacade {
    protected GollekDlLanguageModelingMetricsFacade() {
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingTokenAccuracyMetric() {
        return CanonicalTrainer.Metrics.causalLanguageModelingTokenAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingTokenAccuracyMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.causalLanguageModelingTokenAccuracy(ignoreIndex);
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenAccuracyMetric() {
        return CanonicalTrainer.Metrics.nextTokenAccuracy();
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenAccuracyMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.nextTokenAccuracy(ignoreIndex);
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingLogLossMetric() {
        return CanonicalTrainer.Metrics.causalLanguageModelingLogLoss();
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingLogLossMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.causalLanguageModelingLogLoss(ignoreIndex);
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingCrossEntropyMetric() {
        return CanonicalTrainer.Metrics.causalLanguageModelingCrossEntropy();
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingCrossEntropyMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.causalLanguageModelingCrossEntropy(ignoreIndex);
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingPerplexityMetric() {
        return CanonicalTrainer.Metrics.causalLanguageModelingPerplexity();
    }

    public static Supplier<CanonicalTrainer.Metric> causalLanguageModelingPerplexityMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.causalLanguageModelingPerplexity(ignoreIndex);
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenPerplexityMetric() {
        return CanonicalTrainer.Metrics.nextTokenPerplexity();
    }

    public static Supplier<CanonicalTrainer.Metric> nextTokenPerplexityMetric(float ignoreIndex) {
        return CanonicalTrainer.Metrics.nextTokenPerplexity(ignoreIndex);
    }
}
