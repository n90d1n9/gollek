package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Scalar loss terms emitted by one GRAM deep-supervision step.
 */
public record GramSupervisionStepLoss(
        int supervisionStep,
        double reconstructionNll,
        double klDivergence,
        double latentProcessRewardLoss,
        double adaptiveComputationLoss,
        Map<String, Object> metadata) {

    public GramSupervisionStepLoss {
        if (supervisionStep < 0) {
            throw new IllegalArgumentException("supervisionStep must be >= 0 but was " + supervisionStep);
        }
        reconstructionNll = requireNonNegativeFinite(reconstructionNll, "reconstructionNll");
        klDivergence = requireNonNegativeFinite(klDivergence, "klDivergence");
        latentProcessRewardLoss = requireNonNegativeFinite(latentProcessRewardLoss, "latentProcessRewardLoss");
        adaptiveComputationLoss = requireNonNegativeFinite(adaptiveComputationLoss, "adaptiveComputationLoss");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public double weightedLoss(GramObjectiveConfig config) {
        return config.reconstructionWeight() * reconstructionNll
                + config.effectiveKlWeight() * klDivergence
                + config.latentProcessRewardWeight() * latentProcessRewardLoss
                + config.adaptiveComputationWeight() * adaptiveComputationLoss;
    }

    private static double requireNonNegativeFinite(double value, String name) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(name + " must be finite and >= 0 but was " + value);
        }
        return value;
    }
}
