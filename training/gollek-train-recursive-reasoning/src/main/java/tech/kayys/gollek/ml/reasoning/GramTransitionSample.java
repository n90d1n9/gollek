package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Sampled stochastic-guidance payload for one GRAM latent transition.
 */
public record GramTransitionSample(
        GramTransitionMode mode,
        GramLatentGaussian prior,
        GramLatentGaussian posterior,
        GramLatentGaussian sampledDistribution,
        Tensor epsilon,
        Tensor latentState,
        Tensor klDivergence,
        Map<String, Object> metadata) {

    public GramTransitionSample {
        mode = Objects.requireNonNull(mode, "mode must not be null");
        prior = Objects.requireNonNull(prior, "prior must not be null");
        sampledDistribution = Objects.requireNonNull(sampledDistribution, "sampledDistribution must not be null");
        epsilon = Objects.requireNonNull(epsilon, "epsilon must not be null");
        latentState = Objects.requireNonNull(latentState, "latentState must not be null");
        if (mode == GramTransitionMode.POSTERIOR && posterior == null) {
            throw new IllegalArgumentException("posterior must not be null for posterior samples");
        }
        if (mode == GramTransitionMode.PRIOR && posterior != null) {
            throw new IllegalArgumentException("posterior must be null for prior samples");
        }
        if (mode == GramTransitionMode.POSTERIOR && klDivergence == null) {
            throw new IllegalArgumentException("klDivergence must not be null for posterior samples");
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
