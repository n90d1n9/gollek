package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Non-KL loss terms evaluated at a deep-supervision terminal state.
 */
public record GramTerminalLossTerms(
        double reconstructionNll,
        double latentProcessRewardLoss,
        double adaptiveComputationLoss,
        Map<String, Object> metadata) {

    public GramTerminalLossTerms {
        reconstructionNll = requireNonNegativeFinite(reconstructionNll, "reconstructionNll");
        latentProcessRewardLoss = requireNonNegativeFinite(latentProcessRewardLoss, "latentProcessRewardLoss");
        adaptiveComputationLoss = requireNonNegativeFinite(adaptiveComputationLoss, "adaptiveComputationLoss");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    private static double requireNonNegativeFinite(double value, String name) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(name + " must be finite and >= 0 but was " + value);
        }
        return value;
    }
}
