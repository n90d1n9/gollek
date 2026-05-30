package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Generic token-prediction view over a recursive rollout.
 */
public record DiscreteRolloutTokenReport(
        List<DiscreteTrajectoryTokenPrediction> predictions,
        int selectedTrajectoryIndex,
        Map<String, Object> metadata) {

    public DiscreteRolloutTokenReport {
        predictions = List.copyOf(Objects.requireNonNull(predictions, "predictions must not be null"));
        if (predictions.isEmpty()) {
            throw new IllegalArgumentException("predictions must not be empty");
        }
        if (selectedTrajectoryIndex < 0 || selectedTrajectoryIndex >= predictions.size()) {
            throw new IllegalArgumentException(
                    "selectedTrajectoryIndex must point to a prediction but was " + selectedTrajectoryIndex);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public DiscreteTrajectoryTokenPrediction selectedPrediction() {
        return predictions.get(selectedTrajectoryIndex);
    }
}
