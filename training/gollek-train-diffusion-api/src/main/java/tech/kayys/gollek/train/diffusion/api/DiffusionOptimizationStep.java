package tech.kayys.gollek.train.diffusion.api;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Applies a backend-specific optimization step for a computed loss tensor.
 */
@FunctionalInterface
public interface DiffusionOptimizationStep {

    void update(Tensor loss);
}
