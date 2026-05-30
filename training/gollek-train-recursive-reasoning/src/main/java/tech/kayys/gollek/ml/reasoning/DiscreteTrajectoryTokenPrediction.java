package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Token prediction captured from the final state of one rollout trajectory.
 */
public record DiscreteTrajectoryTokenPrediction(
        int trajectoryIndex,
        int sampleIndex,
        String stateId,
        int[] tokens,
        Map<String, Object> metadata) {

    public DiscreteTrajectoryTokenPrediction {
        if (trajectoryIndex < 0) {
            throw new IllegalArgumentException("trajectoryIndex must be >= 0 but was " + trajectoryIndex);
        }
        if (sampleIndex < 0) {
            throw new IllegalArgumentException("sampleIndex must be >= 0 but was " + sampleIndex);
        }
        stateId = Objects.requireNonNull(stateId, "stateId must not be null");
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }
}
