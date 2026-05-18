package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Compatibility adapters for the historical CanonicalTrainer metric aliases.
 */
@SuppressWarnings("deprecation")
final class TrainerLegacyMetrics {
    private TrainerLegacyMetrics() {
    }

    static Supplier<CanonicalTrainer.Metric> legacy(Supplier<? extends TrainingMetric> factory) {
        return () -> metric(Objects.requireNonNull(factory, "metric factory must not be null").get());
    }

    static CanonicalTrainer.Metric metric(TrainingMetric metric) {
        TrainingMetric resolved = Objects.requireNonNull(metric, "metric factory returned null");
        if (resolved instanceof DetailedTrainingMetric detailedMetric) {
            return new DetailedMetricAdapter(detailedMetric);
        }
        return new MetricAdapter(resolved);
    }

    private record MetricAdapter(TrainingMetric delegate) implements CanonicalTrainer.Metric {
        private MetricAdapter {
            Objects.requireNonNull(delegate, "delegate must not be null");
        }

        @Override
        public String name() {
            return delegate.name();
        }

        @Override
        public void reset() {
            delegate.reset();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            delegate.update(predictions, targets);
        }

        @Override
        public double value() {
            return delegate.value();
        }
    }

    private record DetailedMetricAdapter(DetailedTrainingMetric delegate) implements CanonicalTrainer.DetailedMetric {
        private DetailedMetricAdapter {
            Objects.requireNonNull(delegate, "delegate must not be null");
        }

        @Override
        public String name() {
            return delegate.name();
        }

        @Override
        public void reset() {
            delegate.reset();
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            delegate.update(predictions, targets);
        }

        @Override
        public double value() {
            return delegate.value();
        }

        @Override
        public Map<String, Object> details() {
            return delegate.details();
        }
    }
}
