package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Per-trajectory N-Queens evaluation derived from a recursive reasoning rollout.
 */
public record NQueensTrajectoryEvaluation(
        int trajectoryIndex,
        int sampleIndex,
        String stateId,
        int[] tokens,
        NQueensTokenDecodeResult decoded,
        NQueensEvaluation evaluation,
        Map<String, Object> metadata) {

    public NQueensTrajectoryEvaluation {
        if (trajectoryIndex < 0) {
            throw new IllegalArgumentException("trajectoryIndex must be >= 0 but was " + trajectoryIndex);
        }
        if (sampleIndex < 0) {
            throw new IllegalArgumentException("sampleIndex must be >= 0 but was " + sampleIndex);
        }
        stateId = Objects.requireNonNull(stateId, "stateId must not be null");
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        decoded = Objects.requireNonNull(decoded, "decoded must not be null");
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
