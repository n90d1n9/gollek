package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Backtracking N-Queens completion solver for benchmark coverage denominators and small datasets.
 */
public final class NQueensSolver {
    private NQueensSolver() {
    }

    public static List<NQueensSolution> solve(NQueensProblem problem) {
        return solve(problem, Integer.MAX_VALUE);
    }

    public static List<NQueensSolution> solve(NQueensProblem problem, int maxSolutions) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireNonNegative(maxSolutions, "maxSolutions");
        if (maxSolutions == 0) {
            return List.of();
        }

        SolverState state = new SolverState(problem, maxSolutions, true);
        state.search(0);
        return List.copyOf(state.solutions);
    }

    public static int count(NQueensProblem problem) {
        return count(problem, Integer.MAX_VALUE);
    }

    public static int count(NQueensProblem problem, int maxCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireNonNegative(maxCount, "maxCount");
        if (maxCount == 0) {
            return 0;
        }

        SolverState state = new SolverState(problem, maxCount, false);
        state.search(0);
        return state.count;
    }

    private static int requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }

    private static final class SolverState {
        private final NQueensProblem problem;
        private final int limit;
        private final boolean collectSolutions;
        private final int[] columnsByRow;
        private final boolean[] occupiedColumns;
        private final boolean[] occupiedRisingDiagonals;
        private final boolean[] occupiedFallingDiagonals;
        private final List<NQueensSolution> solutions = new ArrayList<>();
        private int count;

        private SolverState(NQueensProblem problem, int limit, boolean collectSolutions) {
            this.problem = problem;
            this.limit = limit;
            this.collectSolutions = collectSolutions;
            this.columnsByRow = new int[problem.size()];
            Arrays.fill(columnsByRow, NQueensProblem.EMPTY);
            this.occupiedColumns = new boolean[problem.size()];
            this.occupiedRisingDiagonals = new boolean[problem.size() * 2 - 1];
            this.occupiedFallingDiagonals = new boolean[problem.size() * 2 - 1];
        }

        private void search(int row) {
            if (count >= limit) {
                return;
            }
            if (row == problem.size()) {
                count++;
                if (collectSolutions) {
                    solutions.add(new NQueensSolution(columnsByRow));
                }
                return;
            }

            if (problem.hasFixedQueen(row)) {
                placeIfSafe(row, problem.fixedColumn(row));
                return;
            }

            for (int column = 0; column < problem.size(); column++) {
                placeIfSafe(row, column);
                if (count >= limit) {
                    return;
                }
            }
        }

        private void placeIfSafe(int row, int column) {
            int risingDiagonal = row + column;
            int fallingDiagonal = row - column + problem.size() - 1;
            if (occupiedColumns[column]
                    || occupiedRisingDiagonals[risingDiagonal]
                    || occupiedFallingDiagonals[fallingDiagonal]) {
                return;
            }

            columnsByRow[row] = column;
            occupiedColumns[column] = true;
            occupiedRisingDiagonals[risingDiagonal] = true;
            occupiedFallingDiagonals[fallingDiagonal] = true;
            search(row + 1);
            occupiedFallingDiagonals[fallingDiagonal] = false;
            occupiedRisingDiagonals[risingDiagonal] = false;
            occupiedColumns[column] = false;
            columnsByRow[row] = NQueensProblem.EMPTY;
        }
    }
}
