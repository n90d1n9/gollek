package tech.kayys.gollek.ml.reasoning;

import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Tensor helper for diagonal Gaussian KL terms in GRAM's variational objective.
 */
public final class GramGaussianKl {
    private GramGaussianKl() {
    }

    public static Tensor meanPosteriorToPrior(
            GramLatentGaussian posterior,
            GramLatentGaussian prior) {
        Tensor delta = posterior.mean().sub(prior.mean());
        Tensor numerator = posterior.logVariance().exp().add(delta.pow(2.0f));
        Tensor varianceRatio = numerator.div(prior.logVariance().exp());
        return prior.logVariance()
                .sub(posterior.logVariance())
                .add(varianceRatio)
                .add(-1.0f)
                .mul(0.5f)
                .mean();
    }
}
