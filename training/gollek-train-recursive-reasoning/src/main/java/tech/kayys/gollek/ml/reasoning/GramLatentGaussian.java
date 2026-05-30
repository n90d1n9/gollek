package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Diagonal Gaussian latent distribution used by GRAM prior and posterior heads.
 */
public record GramLatentGaussian(Tensor mean, Tensor logVariance) {

    public GramLatentGaussian {
        mean = Objects.requireNonNull(mean, "mean must not be null");
        logVariance = Objects.requireNonNull(logVariance, "logVariance must not be null");
        if (!Arrays.equals(mean.shape().dims(), logVariance.shape().dims())) {
            throw new IllegalArgumentException(
                    "mean and logVariance shapes must match: "
                            + mean.shape()
                            + " vs "
                            + logVariance.shape());
        }
    }
}
