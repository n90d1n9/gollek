package tech.kayys.gollek.trainer.api;

/**
 * Minimal runtime view exposed by long-running trainer sessions.
 */
public interface TrainerSession extends AutoCloseable {

    int currentEpoch();

    int globalStep();

    TrainerConfig config();

    TrainingSummary summary();

    boolean isStopped();

    void stop();

    @Override
    void close();
}
