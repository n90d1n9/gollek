package tech.kayys.gollek.ml;

import tech.kayys.gollek.ml.train.CanonicalTrainer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Base metric registry shared by training option builders.
 */
@SuppressWarnings("deprecation")
public abstract class GollekTrainingMetricRegistrySupport<B extends GollekTrainingMetricRegistrySupport<B>>
        extends GollekTrainingLossBuilderSupport<B> {
    protected final List<Supplier<? extends CanonicalTrainer.Metric>> metricFactories = new ArrayList<>();

    public B metric(Supplier<? extends CanonicalTrainer.Metric> metricFactory) {
        this.metricFactories.add(Objects.requireNonNull(metricFactory, "metric factory must not be null"));
        return self();
    }

    public B metrics(List<? extends Supplier<? extends CanonicalTrainer.Metric>> metrics) {
        if (metrics == null) {
            return self();
        }
        for (Supplier<? extends CanonicalTrainer.Metric> metric : metrics) {
            metric(metric);
        }
        return self();
    }
}
