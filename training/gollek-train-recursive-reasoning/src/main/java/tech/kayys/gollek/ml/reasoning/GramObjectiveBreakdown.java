package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Aggregated GRAM objective terms suitable for reports, checkpoints, and dashboards.
 */
public record GramObjectiveBreakdown(
        List<GramSupervisionStepLoss> supervisionSteps,
        double reconstructionNll,
        double klDivergence,
        double latentProcessRewardLoss,
        double adaptiveComputationLoss,
        double totalLoss,
        GramObjectiveConfig config,
        Map<String, Object> metadata) {

    public GramObjectiveBreakdown {
        supervisionSteps = List.copyOf(Objects.requireNonNull(supervisionSteps, "supervisionSteps must not be null"));
        if (supervisionSteps.isEmpty()) {
            throw new IllegalArgumentException("supervisionSteps must not be empty");
        }
        reconstructionNll = requireFinite(reconstructionNll, "reconstructionNll");
        klDivergence = requireFinite(klDivergence, "klDivergence");
        latentProcessRewardLoss = requireFinite(latentProcessRewardLoss, "latentProcessRewardLoss");
        adaptiveComputationLoss = requireFinite(adaptiveComputationLoss, "adaptiveComputationLoss");
        totalLoss = requireFinite(totalLoss, "totalLoss");
        config = Objects.requireNonNull(config, "config must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    private static double requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite but was " + value);
        }
        return value;
    }
}
