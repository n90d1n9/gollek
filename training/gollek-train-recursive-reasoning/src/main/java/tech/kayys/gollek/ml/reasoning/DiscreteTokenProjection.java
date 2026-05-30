package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Generic projected token payload to attach to recursive states.
 */
public record DiscreteTokenProjection(
        int[] tokens,
        String projectionMode,
        Map<String, Object> metadata) {

    public DiscreteTokenProjection {
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        projectionMode = Objects.requireNonNull(projectionMode, "projectionMode must not be null");
        if (projectionMode.isBlank()) {
            throw new IllegalArgumentException("projectionMode must not be blank");
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }
}
