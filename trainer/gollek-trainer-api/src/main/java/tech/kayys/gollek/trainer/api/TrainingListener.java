package tech.kayys.gollek.trainer.api;

/**
 * Canonical trainer lifecycle contract for orchestration, logging, and
 * checkpoint integrations.
 */
public interface TrainingListener extends AutoCloseable {

    default void onTrainingStart(TrainerSession session) {
    }

    default void onEpochStart(TrainerSession session, int epoch) {
    }

    default void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
    }

    default void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
    }

    default void onBatchStart(TrainerSession session, int step) {
    }

    default void onBatchEnd(TrainerSession session, int step, double loss) {
    }

    default void onEarlyStopping(TrainerSession session, int epoch) {
    }

    default void onTrainingError(TrainerSession session, Exception error) {
    }

    default void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
    }

    @Override
    default void close() {
        // Default no-op for listeners without external resources.
    }
}
