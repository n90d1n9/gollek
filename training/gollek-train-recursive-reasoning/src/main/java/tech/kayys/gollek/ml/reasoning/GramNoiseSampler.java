package tech.kayys.gollek.ml.reasoning;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Provides epsilon noise for GRAM's reparameterized latent sampling.
 */
@FunctionalInterface
public interface GramNoiseSampler {
    Tensor sampleEpsilon(
            GramLatentGaussian distribution,
            GramTransitionInput input);
}
