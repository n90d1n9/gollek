package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Output of projecting model logits into paper-style N-Queens board tokens.
 */
public record NQueensTokenProjectionResult(
        int[] tokens,
        NQueensTokenProjectionMode mode,
        Map<String, Object> metadata) {

    public NQueensTokenProjectionResult {
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        mode = Objects.requireNonNull(mode, "mode must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }
}
