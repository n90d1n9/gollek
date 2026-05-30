package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;

/**
 * Candidate N-Queens solution with one column per row; -1 means no queen in that row.
 */
public record NQueensSolution(int[] columnsByRow) {

    public NQueensSolution {
        if (columnsByRow == null) {
            throw new NullPointerException("columnsByRow must not be null");
        }
        if (columnsByRow.length < 1) {
            throw new IllegalArgumentException("columnsByRow must not be empty");
        }
        columnsByRow = NQueensProblem.requireColumns(columnsByRow.length, columnsByRow, "columnsByRow");
    }

    public static NQueensSolution ofColumns(int... columnsByRow) {
        return new NQueensSolution(columnsByRow);
    }

    @Override
    public int[] columnsByRow() {
        return columnsByRow.clone();
    }

    public int size() {
        return columnsByRow.length;
    }

    public int column(int row) {
        return columnsByRow[row];
    }

    public boolean complete() {
        for (int column : columnsByRow) {
            if (column == NQueensProblem.EMPTY) {
                return false;
            }
        }
        return true;
    }

    public int[] toTokens() {
        return NQueensProblem.boardTokens(columnsByRow);
    }

    String canonicalKey() {
        return Arrays.toString(columnsByRow);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof NQueensSolution solution
                && Arrays.equals(columnsByRow, solution.columnsByRow);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(columnsByRow);
    }
}
