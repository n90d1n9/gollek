package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Aggregates GRAM deep-supervision terms into a single optimization objective.
 */
public final class GramObjective {
    private GramObjective() {
    }

    public static GramObjectiveBreakdown evaluate(
            List<GramSupervisionStepLoss> supervisionSteps,
            GramObjectiveConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        List<GramSupervisionStepLoss> steps = List.copyOf(
                Objects.requireNonNull(supervisionSteps, "supervisionSteps must not be null"));
        if (steps.isEmpty()) {
            throw new IllegalArgumentException("supervisionSteps must not be empty");
        }

        double reconstructionNll = 0.0;
        double klDivergence = 0.0;
        double latentProcessRewardLoss = 0.0;
        double adaptiveComputationLoss = 0.0;
        double totalLoss = 0.0;

        for (GramSupervisionStepLoss step : steps) {
            reconstructionNll += step.reconstructionNll();
            klDivergence += step.klDivergence();
            latentProcessRewardLoss += step.latentProcessRewardLoss();
            adaptiveComputationLoss += step.adaptiveComputationLoss();
            totalLoss += step.weightedLoss(config);
        }

        if (config.reduction() == GramObjectiveReduction.MEAN) {
            double count = steps.size();
            reconstructionNll /= count;
            klDivergence /= count;
            latentProcessRewardLoss /= count;
            adaptiveComputationLoss /= count;
            totalLoss /= count;
        }

        return new GramObjectiveBreakdown(
                steps,
                reconstructionNll,
                klDivergence,
                latentProcessRewardLoss,
                adaptiveComputationLoss,
                totalLoss,
                config,
                Map.of(
                        "objective", "gram-surrogate-elbo",
                        "klBalance", config.klBalance(),
                        "stepCount", steps.size(),
                        "truncatedSurrogate", config.truncatedSurrogate()));
    }
}
