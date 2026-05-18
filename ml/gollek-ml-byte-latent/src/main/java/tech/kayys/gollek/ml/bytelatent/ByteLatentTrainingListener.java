package tech.kayys.gollek.ml.bytelatent;

import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Listener contract for byte-latent trainer sessions.
 */
public interface ByteLatentTrainingListener extends AutoCloseable {

    default void onTrainingStart(ByteLatentTrainerSession session) {
    }

    default void onEpochStart(ByteLatentTrainerSession session, int epoch) {
    }

    default void onEpochEnd(ByteLatentTrainerSession session, int epoch, double trainLoss) {
    }

    default void onBatchStart(ByteLatentTrainerSession session, int step) {
    }

    default void onBatchEnd(ByteLatentTrainerSession session, int step, double loss) {
    }

    default void onTrainingError(ByteLatentTrainerSession session, Exception error) {
    }

    default void onTrainingEnd(ByteLatentTrainerSession session, TrainingSummary summary) {
    }

    @Override
    default void close() {
    }
}
