package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class TrainingMetricBuilderTest {

    @Test
    void customMetricFactoryCreatesIndependentPhaseState() {
        Supplier<TrainingMetric> factory = TrainingMetrics.custom("mean_bias", BiasState::new)
                .reset(BiasState::reset)
                .update(BiasState::update)
                .value(BiasState::meanBias)
                .build();
        TrainingMetric trainMetric = factory.get();
        TrainingMetric validationMetric = factory.get();

        trainMetric.update(
                GradTensor.of(new float[] {3f, 5f}, 2),
                GradTensor.of(new float[] {1f, 1f}, 2));
        validationMetric.update(
                GradTensor.of(new float[] {2f}, 1),
                GradTensor.of(new float[] {1f}, 1));

        assertEquals("mean_bias", trainMetric.name());
        assertEquals(3.0, trainMetric.value(), 1e-6);
        assertEquals(1.0, validationMetric.value(), 1e-6);

        trainMetric.reset();

        assertEquals(0.0, trainMetric.value(), 1e-6);
        assertEquals(1.0, validationMetric.value(), 1e-6);
    }

    @Test
    void customMetricCanExposeDetails() {
        Supplier<TrainingMetric> factory = TrainingMetrics.custom("mean_bias", BiasState::new)
                .update(BiasState::update)
                .value(BiasState::meanBias)
                .details(state -> Map.of(
                        "count", state.count,
                        "sumBias", state.sumBias))
                .build();
        DetailedTrainingMetric metric = assertInstanceOf(DetailedTrainingMetric.class, factory.get());

        metric.update(
                GradTensor.of(new float[] {4f, 6f}, 2),
                GradTensor.of(new float[] {1f, 2f}, 2));

        assertEquals(3.5, metric.value(), 1e-6);
        assertEquals(Map.of("count", 2L, "sumBias", 7.0), metric.details());
    }

    @Test
    void customMetricValidatesRequiredConfigurationAndState() {
        assertThrows(IllegalArgumentException.class, () -> TrainingMetrics.custom(" ", BiasState::new));

        TrainingMetricBuilder<BiasState> missingUpdate = TrainingMetrics.custom("missing_update", BiasState::new)
                .value(BiasState::meanBias);
        assertEquals(
                "metric update callback must be configured",
                assertThrows(NullPointerException.class, missingUpdate::build).getMessage());

        TrainingMetricBuilder<BiasState> missingValue = TrainingMetrics.custom("missing_value", BiasState::new)
                .update(BiasState::update);
        assertEquals(
                "metric value callback must be configured",
                assertThrows(NullPointerException.class, missingValue::build).getMessage());

        Supplier<TrainingMetric> nullStateFactory = TrainingMetrics.<BiasState>custom("null_state", () -> null)
                .update(BiasState::update)
                .value(BiasState::meanBias)
                .build();
        assertEquals(
                "metric state factory returned null",
                assertThrows(NullPointerException.class, nullStateFactory::get).getMessage());
    }

    private static final class BiasState {
        private double sumBias;
        private long count;

        private void reset() {
            sumBias = 0.0;
            count = 0L;
        }

        private void update(GradTensor predictions, GradTensor targets) {
            float[] predictionValues = predictions.data();
            float[] targetValues = targets.data();
            if (predictionValues.length != targetValues.length) {
                throw new IllegalArgumentException("prediction and target length must match");
            }
            for (int i = 0; i < predictionValues.length; i++) {
                sumBias += predictionValues[i] - targetValues[i];
                count++;
            }
        }

        private double meanBias() {
            return count == 0L ? 0.0 : sumBias / count;
        }
    }
}
