package tech.kayys.gollek.train.diffusion.api;

import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Minimal runtime view for a diffusion OPD training session.
 *
 * <p>This interface models the Java-side lifecycle for the DiffusionOPD
 * training recipe from arXiv:2605.15055.
 */
public interface DiffusionOpdSession extends AutoCloseable {

    DiffusionOpdConfig config();

    TrainingSummary fit();

    TrainingSummary summary();

    boolean isStopped();

    void stop();

    @Override
    default void close() {
        stop();
    }
}
