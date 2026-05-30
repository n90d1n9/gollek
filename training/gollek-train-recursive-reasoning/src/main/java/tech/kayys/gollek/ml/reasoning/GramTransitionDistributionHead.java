package tech.kayys.gollek.ml.reasoning;

/**
 * Produces a diagonal Gaussian transition distribution from a deterministic proposal.
 */
@FunctionalInterface
public interface GramTransitionDistributionHead {
    GramLatentGaussian distribution(GramTransitionInput input);
}
