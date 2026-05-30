package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Backtracking graph-coloring solver for exact coverage denominators and small datasets.
 */
public final class GraphColoringSolver {
    private GraphColoringSolver() {
    }

    public static List<GraphColoringSolution> solve(GraphColoringProblem problem) {
        return solve(problem, Integer.MAX_VALUE);
    }

    public static List<GraphColoringSolution> solve(GraphColoringProblem problem, int maxSolutions) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireNonNegative(maxSolutions, "maxSolutions");
        if (maxSolutions == 0) {
            return List.of();
        }

        SolverState state = new SolverState(problem, maxSolutions, true);
        state.search(0);
        return List.copyOf(state.solutions);
    }

    public static int count(GraphColoringProblem problem) {
        return count(problem, Integer.MAX_VALUE);
    }

    public static int count(GraphColoringProblem problem, int maxCount) {
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
        private final GraphColoringProblem problem;
        private final int limit;
        private final boolean collectSolutions;
        private final int[] colorsByNode;
        private final List<GraphColoringSolution> solutions = new ArrayList<>();
        private int count;

        private SolverState(GraphColoringProblem problem, int limit, boolean collectSolutions) {
            this.problem = problem;
            this.limit = limit;
            this.collectSolutions = collectSolutions;
            this.colorsByNode = new int[problem.nodeCount()];
            Arrays.fill(colorsByNode, GraphColoringProblem.UNCOLORED);
        }

        private void search(int node) {
            if (count >= limit) {
                return;
            }
            if (node == problem.nodeCount()) {
                count++;
                if (collectSolutions) {
                    solutions.add(new GraphColoringSolution(problem.colorCount(), colorsByNode));
                }
                return;
            }

            if (problem.hasFixedColor(node)) {
                placeIfSafe(node, problem.fixedColor(node));
                return;
            }

            for (int color = 0; color < problem.colorCount(); color++) {
                placeIfSafe(node, color);
                if (count >= limit) {
                    return;
                }
            }
        }

        private void placeIfSafe(int node, int color) {
            if (!safe(node, color)) {
                return;
            }
            colorsByNode[node] = color;
            search(node + 1);
            colorsByNode[node] = GraphColoringProblem.UNCOLORED;
        }

        private boolean safe(int node, int color) {
            for (GraphColoringEdge edge : problem.edges()) {
                int neighbor = -1;
                if (edge.leftNode() == node) {
                    neighbor = edge.rightNode();
                } else if (edge.rightNode() == node) {
                    neighbor = edge.leftNode();
                }
                if (neighbor >= 0 && colorsByNode[neighbor] == color) {
                    return false;
                }
            }
            return true;
        }
    }
}
