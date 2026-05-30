package tech.kayys.gollek.ml.reasoning;

import java.util.Map;

/**
 * Validation report for one N-Queens candidate.
 */
public record NQueensEvaluation(
        boolean complete,
        boolean respectsFixedQueens,
        boolean uniqueColumns,
        boolean diagonalSafe,
        int missingRowCount,
        int fixedViolationCount,
        int columnConflictCount,
        int diagonalConflictCount,
        int conflictCount,
        Map<String, Object> metadata) {

    public NQueensEvaluation {
        missingRowCount = requireNonNegative(missingRowCount, "missingRowCount");
        fixedViolationCount = requireNonNegative(fixedViolationCount, "fixedViolationCount");
        columnConflictCount = requireNonNegative(columnConflictCount, "columnConflictCount");
        diagonalConflictCount = requireNonNegative(diagonalConflictCount, "diagonalConflictCount");
        conflictCount = requireNonNegative(conflictCount, "conflictCount");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public boolean valid() {
        return complete && respectsFixedQueens && uniqueColumns && diagonalSafe && conflictCount == 0;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }
}
