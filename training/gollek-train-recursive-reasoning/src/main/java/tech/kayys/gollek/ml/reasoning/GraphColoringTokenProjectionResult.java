package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Output of projecting model logits into graph-coloring tokens.
 */
public record GraphColoringTokenProjectionResult(
        int[] tokens,
        GraphColoringTokenProjectionMode mode,
        Map<String, Object> metadata) {

    public GraphColoringTokenProjectionResult {
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        mode = Objects.requireNonNull(mode, "mode must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }
}
