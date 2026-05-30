package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;

/**
 * N-Queens puzzle input with optional fixed queens, one row per entry.
 */
public record NQueensProblem(int size, int[] fixedColumnsByRow) {
    public static final int EMPTY = -1;
    public static final int EMPTY_TOKEN = 1;
    public static final int QUEEN_TOKEN = 2;

    public NQueensProblem {
        if (size < 1) {
            throw new IllegalArgumentException("size must be >= 1 but was " + size);
        }
        fixedColumnsByRow = requireColumns(size, fixedColumnsByRow, "fixedColumnsByRow");
        requireNoFixedConflicts(fixedColumnsByRow);
    }

    public static NQueensProblem empty(int size) {
        int[] columns = new int[size];
        Arrays.fill(columns, EMPTY);
        return new NQueensProblem(size, columns);
    }

    public static NQueensProblem ofFixedColumns(int... fixedColumnsByRow) {
        return new NQueensProblem(fixedColumnsByRow.length, fixedColumnsByRow);
    }

    @Override
    public int[] fixedColumnsByRow() {
        return fixedColumnsByRow.clone();
    }

    public boolean hasFixedQueen(int row) {
        return fixedColumnsByRow[row] != EMPTY;
    }

    public int fixedColumn(int row) {
        return fixedColumnsByRow[row];
    }

    public int[] toTokens() {
        return boardTokens(fixedColumnsByRow);
    }

    static int[] requireColumns(int size, int[] columns, String name) {
        if (columns == null) {
            throw new NullPointerException(name + " must not be null");
        }
        if (columns.length != size) {
            throw new IllegalArgumentException(name + " length must be " + size + " but was " + columns.length);
        }
        int[] copy = columns.clone();
        for (int row = 0; row < copy.length; row++) {
            int column = copy[row];
            if (column < EMPTY || column >= size) {
                throw new IllegalArgumentException(
                        name + "[" + row + "] must be -1 or in [0, " + (size - 1) + "] but was " + column);
            }
        }
        return copy;
    }

    static int[] boardTokens(int[] columnsByRow) {
        int size = columnsByRow.length;
        int[] tokens = new int[size * size];
        Arrays.fill(tokens, EMPTY_TOKEN);
        for (int row = 0; row < size; row++) {
            int column = columnsByRow[row];
            if (column != EMPTY) {
                tokens[row * size + column] = QUEEN_TOKEN;
            }
        }
        return tokens;
    }

    private static void requireNoFixedConflicts(int[] fixedColumnsByRow) {
        for (int leftRow = 0; leftRow < fixedColumnsByRow.length; leftRow++) {
            int leftColumn = fixedColumnsByRow[leftRow];
            if (leftColumn == EMPTY) {
                continue;
            }
            for (int rightRow = leftRow + 1; rightRow < fixedColumnsByRow.length; rightRow++) {
                int rightColumn = fixedColumnsByRow[rightRow];
                if (rightColumn == EMPTY) {
                    continue;
                }
                if (leftColumn == rightColumn
                        || Math.abs(leftRow - rightRow) == Math.abs(leftColumn - rightColumn)) {
                    throw new IllegalArgumentException(
                            "fixed queens conflict at rows " + leftRow + " and " + rightRow);
                }
            }
        }
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof NQueensProblem problem
                && size == problem.size
                && Arrays.equals(fixedColumnsByRow, problem.fixedColumnsByRow);
    }

    @Override
    public int hashCode() {
        return 31 * Integer.hashCode(size) + Arrays.hashCode(fixedColumnsByRow);
    }
}
