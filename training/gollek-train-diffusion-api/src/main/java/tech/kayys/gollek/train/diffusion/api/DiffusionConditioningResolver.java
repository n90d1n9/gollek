package tech.kayys.gollek.train.diffusion.api;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Resolves model conditioning tensors for a prompt sample.
 */
@FunctionalInterface
public interface DiffusionConditioningResolver {

    Tensor resolve(DiffusionPromptSample sample);
}
