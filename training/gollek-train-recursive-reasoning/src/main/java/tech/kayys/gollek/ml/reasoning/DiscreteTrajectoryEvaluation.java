package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Benchmark-neutral evaluation for one rollout trajectory token prediction.
 */
public record DiscreteTrajectoryEvaluation(
        int trajectoryIndex,
        int sampleIndex,
        String stateId,
        int[] tokens,
        DiscreteTokenEvaluation evaluation,
        Map<String, Object> metadata) {

    public DiscreteTrajectoryEvaluation {
        if (trajectoryIndex < 0) {
            throw new IllegalArgumentException("trajectoryIndex must be >= 0 but was " + trajectoryIndex);
        }
        if (sampleIndex < 0) {
            throw new IllegalArgumentException("sampleIndex must be >= 0 but was " + sampleIndex);
        }
        stateId = Objects.requireNonNull(stateId, "stateId must not be null");
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        evaluation = Objects.requireNonNull(evaluation, "evaluation must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }

    public boolean valid() {
        return evaluation.valid();
    }
}
