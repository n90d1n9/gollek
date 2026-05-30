package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Reparameterized diagonal Gaussian sampling helper: z = mean + std * epsilon.
 */
public final class GramReparameterization {
    private GramReparameterization() {
    }

    public static Tensor sample(GramLatentGaussian distribution, Tensor epsilon) {
        Objects.requireNonNull(distribution, "distribution must not be null");
        Objects.requireNonNull(epsilon, "epsilon must not be null");
        if (!Arrays.equals(distribution.mean().shape().dims(), epsilon.shape().dims())) {
            throw new IllegalArgumentException(
                    "epsilon shape must match distribution mean shape: "
                            + epsilon.shape()
                            + " vs "
                            + distribution.mean().shape());
        }
        Tensor standardDeviation = distribution.logVariance().mul(0.5f).exp();
        return distribution.mean().add(standardDeviation.mul(epsilon));
    }
}
