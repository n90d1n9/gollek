package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Diagnostic result of decoding board tokens into a row-column N-Queens candidate.
 */
public record NQueensTokenDecodeResult(
        NQueensSolution solution,
        int invalidTokenCount,
        int emptyRowCount,
        int multiQueenRowCount,
        int queenTokenCount,
        Map<String, Object> metadata) {

    public NQueensTokenDecodeResult {
        solution = Objects.requireNonNull(solution, "solution must not be null");
        invalidTokenCount = requireNonNegative(invalidTokenCount, "invalidTokenCount");
        emptyRowCount = requireNonNegative(emptyRowCount, "emptyRowCount");
        multiQueenRowCount = requireNonNegative(multiQueenRowCount, "multiQueenRowCount");
        queenTokenCount = requireNonNegative(queenTokenCount, "queenTokenCount");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public boolean clean() {
        return invalidTokenCount == 0 && emptyRowCount == 0 && multiQueenRowCount == 0;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }
}
