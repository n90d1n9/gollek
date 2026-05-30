package tech.kayys.gollek.ml.reasoning;

import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Minimal runtime session for recursive reasoning trainer flows.
 */
public interface RecursiveReasoningSession extends AutoCloseable {

    RecursiveReasoningConfig config();

    int currentSupervisionStep();

    int currentTransitionIndex();

    int activeSampleCount();

    TrainingSummary fit();

    RecursiveReasoningRolloutSummary rolloutSummary();

    boolean isStopped();

    void stop();

    @Override
    default void close() {
        stop();
    }
}
