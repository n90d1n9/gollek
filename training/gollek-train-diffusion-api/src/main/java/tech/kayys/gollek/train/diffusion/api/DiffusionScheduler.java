package tech.kayys.gollek.train.diffusion.api;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Java-first scheduler contract for diffusion training loops.
 */
public interface DiffusionScheduler {

    Tensor step(Tensor xT, Tensor modelPrediction, int timestepIndex);

    int[] timesteps();
}
