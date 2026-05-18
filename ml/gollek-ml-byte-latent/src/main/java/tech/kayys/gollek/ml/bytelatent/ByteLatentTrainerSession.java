package tech.kayys.gollek.ml.bytelatent;

import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Minimal runtime session for byte-latent trainer flows.
 */
public interface ByteLatentTrainerSession extends AutoCloseable {

    ByteLatentTrainerConfig config();

    int currentEpoch();

    int globalStep();

    TrainingSummary fit();

    TrainingSummary summary();

    boolean isStopped();

    void stop();

    @Override
    default void close() {
        stop();
    }
}
