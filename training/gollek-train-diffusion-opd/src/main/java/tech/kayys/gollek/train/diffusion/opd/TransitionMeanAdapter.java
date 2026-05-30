package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Converts model outputs into scheduler-aware transition means.
 *
 * <p>This abstraction exists because DiffusionOPD supervises transition means
 * instead of raw backend-specific denoiser outputs. See arXiv:2605.15055.
 */
public interface TransitionMeanAdapter {

    /**
     * Converts the current noisy state and model prediction into the scheduler-step mean that OPD
     * supervises at the given timestep.
     */
    Tensor transitionMean(Tensor xT, Tensor modelPrediction, int timestepIndex);

    /**
     * Returns the effective variance associated with the adapted scheduler step at the given
     * timestep.
     */
    float stepVariance(int timestepIndex);
}
