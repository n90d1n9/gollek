package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Validation report for one graph-coloring candidate.
 */
public record GraphColoringEvaluation(
        boolean complete,
        boolean respectsFixedColors,
        boolean edgeSafe,
        int uncoloredNodeCount,
        int fixedViolationCount,
        int edgeConflictCount,
        int invalidTokenCount,
        int conflictCount,
        Map<String, Object> metadata) {

    public GraphColoringEvaluation {
        uncoloredNodeCount = requireNonNegative(uncoloredNodeCount, "uncoloredNodeCount");
        fixedViolationCount = requireNonNegative(fixedViolationCount, "fixedViolationCount");
        edgeConflictCount = requireNonNegative(edgeConflictCount, "edgeConflictCount");
        invalidTokenCount = requireNonNegative(invalidTokenCount, "invalidTokenCount");
        conflictCount = requireNonNegative(conflictCount, "conflictCount");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public boolean valid() {
        return complete
                && respectsFixedColors
                && edgeSafe
                && invalidTokenCount == 0
                && conflictCount == 0;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }
}
