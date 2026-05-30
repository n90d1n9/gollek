package tech.kayys.gollek.ml.reasoning;

import java.util.Objects;
import tech.kayys.gollek.core.tensor.Tensor;

/**
 * Tensor-level weighted GRAM objective for autograd-backed trainers.
 */
public final class GramTensorObjective {
    private GramTensorObjective() {
    }

    public static Tensor loss(
            Tensor reconstructionNll,
            Tensor klDivergence,
            GramObjectiveConfig config) {
        return loss(reconstructionNll, klDivergence, null, null, config);
    }

    public static Tensor loss(
            Tensor reconstructionNll,
            Tensor klDivergence,
            Tensor latentProcessRewardLoss,
            Tensor adaptiveComputationLoss,
            GramObjectiveConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        Tensor total = Objects.requireNonNull(reconstructionNll, "reconstructionNll must not be null")
                .mul((float) config.reconstructionWeight());
        total = total.add(Objects.requireNonNull(klDivergence, "klDivergence must not be null")
                .mul((float) config.effectiveKlWeight()));
        if (latentProcessRewardLoss != null && config.latentProcessRewardWeight() > 0.0) {
            total = total.add(latentProcessRewardLoss.mul((float) config.latentProcessRewardWeight()));
        }
        if (adaptiveComputationLoss != null && config.adaptiveComputationWeight() > 0.0) {
            total = total.add(adaptiveComputationLoss.mul((float) config.adaptiveComputationWeight()));
        }
        return total;
    }
}
