package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Diagnostic result of decoding graph-coloring tokens into color assignments.
 */
public record GraphColoringTokenDecodeResult(
        GraphColoringSolution solution,
        int invalidTokenCount,
        int uncoloredNodeCount,
        int colorTokenCount,
        Map<String, Object> metadata) {

    public GraphColoringTokenDecodeResult {
        solution = Objects.requireNonNull(solution, "solution must not be null");
        invalidTokenCount = requireNonNegative(invalidTokenCount, "invalidTokenCount");
        uncoloredNodeCount = requireNonNegative(uncoloredNodeCount, "uncoloredNodeCount");
        colorTokenCount = requireNonNegative(colorTokenCount, "colorTokenCount");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public boolean clean() {
        return invalidTokenCount == 0 && uncoloredNodeCount == 0;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }
}
