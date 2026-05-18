package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * Bridges metric snapshot validation failures into the trainer failure state.
 */
final class TrainerMetricFailureRecorder implements TrainerMetricSnapshots.FailureRecorder {
    private final TrainerFailureState failureState;

    TrainerMetricFailureRecorder(TrainerFailureState failureState) {
        this.failureState = Objects.requireNonNull(failureState, "failureState must not be null");
    }

    @Override
    public String invalidValue(
            String phase,
            String metricName,
            double value,
            boolean optimizerStepSkipped) {
        return failureState.recordInvalidMetricValue(phase, metricName, value, optimizerStepSkipped);
    }

    @Override
    public String invalidDetail(String phase, String metricName, String detailPath, double value) {
        return failureState.recordInvalidMetricDetail(phase, metricName, detailPath, value);
    }

    @Override
    public String invalidName(String phase, String metricName) {
        return failureState.recordInvalidMetricName(phase, metricName);
    }

    @Override
    public String invalidFailure(
            String phase,
            String metricName,
            String kind,
            String detailPath,
            RuntimeException error) {
        return failureState.recordInvalidMetricFailure(phase, metricName, kind, detailPath, error);
    }
}
