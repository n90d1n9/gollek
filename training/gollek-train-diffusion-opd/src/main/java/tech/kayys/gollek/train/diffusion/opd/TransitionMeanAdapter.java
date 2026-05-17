package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Converts model outputs into scheduler-aware transition means.
 *
 * <p>This abstraction exists because DiffusionOPD supervises transition means
 * instead of raw backend-specific denoiser outputs. See arXiv:2605.15055.
 */
public interface TransitionMeanAdapter {

    Tensor transitionMean(Tensor xT, Tensor modelPrediction, int timestepIndex);

    float stepVariance(int timestepIndex);
}
