package tech.kayys.gollek.diffusion.scheduler;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * SCHEDULER (FINAL, CORRECT DDIM)
 */
public interface Scheduler {
    Tensor step(Tensor x_t, Tensor eps, int tIndex);

    int[] timesteps();
}