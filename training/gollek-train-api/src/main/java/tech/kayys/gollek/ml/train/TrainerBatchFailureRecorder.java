package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * Bridges batch and tensor guard failures into the trainer failure state.
 */
final class TrainerBatchFailureRecorder implements TrainerBatchGuards.FailureRecorder {
    private final TrainerFailureState failureState;
    private final Runnable discardPendingGradients;

    TrainerBatchFailureRecorder(TrainerFailureState failureState, Runnable discardPendingGradients) {
        this.failureState = Objects.requireNonNull(failureState, "failureState must not be null");
        this.discardPendingGradients = Objects.requireNonNull(
                discardPendingGradients,
                "discardPendingGradients must not be null");
    }

    @Override
    public String invalidBatch(
            String phase,
            String reason,
            String message,
            boolean optimizerStepSkipped) {
        return failureState.recordInvalidBatch(phase, reason, message, optimizerStepSkipped);
    }

    @Override
    public String invalidLossShape(
            String phase,
            String shape,
            long elements,
            boolean optimizerStepSkipped) {
        return failureState.recordInvalidLossShape(phase, shape, elements, optimizerStepSkipped);
    }

    @Override
    public String nonFinite(
            String phase,
            String kind,
            double value,
            String label,
            boolean optimizerStepSkipped) {
        return failureState.recordNonFinite(phase, kind, value, label, optimizerStepSkipped);
    }

    @Override
    public void discardPendingGradients() {
        discardPendingGradients.run();
    }
}
