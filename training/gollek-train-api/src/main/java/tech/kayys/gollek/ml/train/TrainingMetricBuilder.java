package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Builder for stateful custom metrics.
 *
 * <p>The returned factory creates a fresh metric state for each trainer phase,
 * so train and validation metrics never accidentally share counters.</p>
 *
 * @param <S> mutable state owned by one metric instance
 */
public final class TrainingMetricBuilder<S> {
    private final String name;
    private final Supplier<S> stateFactory;
    private Consumer<S> reset = state -> {
    };
    private TrainingMetricUpdater<S> update;
    private ToDoubleFunction<S> value;
    private Function<S, Map<String, Object>> details;

    TrainingMetricBuilder(String name, Supplier<S> stateFactory) {
        this.name = TrainerMetricSnapshots.requireName(name);
        this.stateFactory = Objects.requireNonNull(stateFactory, "metric state factory must not be null");
    }

    public TrainingMetricBuilder<S> reset(Consumer<S> reset) {
        this.reset = Objects.requireNonNull(reset, "metric reset callback must not be null");
        return this;
    }

    public TrainingMetricBuilder<S> update(TrainingMetricUpdater<S> update) {
        this.update = Objects.requireNonNull(update, "metric update callback must not be null");
        return this;
    }

    public TrainingMetricBuilder<S> value(ToDoubleFunction<S> value) {
        this.value = Objects.requireNonNull(value, "metric value callback must not be null");
        return this;
    }

    public TrainingMetricBuilder<S> details(Function<S, Map<String, Object>> details) {
        this.details = Objects.requireNonNull(details, "metric details callback must not be null");
        return this;
    }

    public Supplier<TrainingMetric> build() {
        TrainingMetricUpdater<S> metricUpdate =
                Objects.requireNonNull(update, "metric update callback must be configured");
        ToDoubleFunction<S> metricValue =
                Objects.requireNonNull(value, "metric value callback must be configured");
        Function<S, Map<String, Object>> metricDetails = details;
        Consumer<S> metricReset = reset;
        if (metricDetails == null) {
            return () -> new ScalarMetric<>(
                    name,
                    newState(),
                    metricReset,
                    metricUpdate,
                    metricValue);
        }
        return () -> new DetailedMetric<>(
                name,
                newState(),
                metricReset,
                metricUpdate,
                metricValue,
                metricDetails);
    }

    private S newState() {
        return Objects.requireNonNull(stateFactory.get(), "metric state factory returned null");
    }

    private static class ScalarMetric<S> implements TrainingMetric {
        private final String name;
        private final S state;
        private final Consumer<S> reset;
        private final TrainingMetricUpdater<S> update;
        private final ToDoubleFunction<S> value;

        private ScalarMetric(
                String name,
                S state,
                Consumer<S> reset,
                TrainingMetricUpdater<S> update,
                ToDoubleFunction<S> value) {
            this.name = name;
            this.state = state;
            this.reset = reset;
            this.update = update;
            this.value = value;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public void reset() {
            reset.accept(state);
        }

        @Override
        public void update(GradTensor predictions, GradTensor targets) {
            update.update(state, predictions, targets);
        }

        @Override
        public double value() {
            return value.applyAsDouble(state);
        }
    }

    private static final class DetailedMetric<S> extends ScalarMetric<S> implements DetailedTrainingMetric {
        private final S state;
        private final Function<S, Map<String, Object>> details;

        private DetailedMetric(
                String name,
                S state,
                Consumer<S> reset,
                TrainingMetricUpdater<S> update,
                ToDoubleFunction<S> value,
                Function<S, Map<String, Object>> details) {
            super(name, state, reset, update, value);
            this.state = state;
            this.details = details;
        }

        @Override
        public Map<String, Object> details() {
            Map<String, Object> snapshot = details.apply(state);
            return snapshot == null ? Map.of() : snapshot;
        }
    }
}
