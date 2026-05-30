package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Backend-neutral projection of dense logits into discrete token ids.
 */
public record DiscreteTokenProjectionResult(
        int[] tokens,
        int itemCount,
        int vocabSize,
        Map<String, Object> metadata) {

    public DiscreteTokenProjectionResult {
        tokens = Objects.requireNonNull(tokens, "tokens must not be null").clone();
        if (itemCount < 1) {
            throw new IllegalArgumentException("itemCount must be >= 1 but was " + itemCount);
        }
        if (vocabSize < 1) {
            throw new IllegalArgumentException("vocabSize must be >= 1 but was " + vocabSize);
        }
        if (tokens.length != itemCount) {
            throw new IllegalArgumentException(
                    "tokens length must match itemCount: " + tokens.length + " vs " + itemCount);
        }
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    @Override
    public int[] tokens() {
        return tokens.clone();
    }
}
