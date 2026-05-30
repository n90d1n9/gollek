package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;

/**
 * Candidate graph coloring; -1 means no color assigned to a node.
 */
public record GraphColoringSolution(int colorCount, int[] colorsByNode) {

    public GraphColoringSolution {
        if (colorCount < 1) {
            throw new IllegalArgumentException("colorCount must be >= 1 but was " + colorCount);
        }
        if (colorsByNode == null) {
            throw new NullPointerException("colorsByNode must not be null");
        }
        if (colorsByNode.length < 1) {
            throw new IllegalArgumentException("colorsByNode must not be empty");
        }
        colorsByNode = GraphColoringProblem.requireColors(
                colorsByNode.length,
                colorCount,
                colorsByNode,
                "colorsByNode");
    }

    public static GraphColoringSolution ofColors(int colorCount, int... colorsByNode) {
        return new GraphColoringSolution(colorCount, colorsByNode);
    }

    @Override
    public int[] colorsByNode() {
        return colorsByNode.clone();
    }

    public int nodeCount() {
        return colorsByNode.length;
    }

    public int color(int node) {
        return colorsByNode[node];
    }

    public boolean complete() {
        for (int color : colorsByNode) {
            if (color == GraphColoringProblem.UNCOLORED) {
                return false;
            }
        }
        return true;
    }

    public int[] toTokens() {
        return GraphColoringProblem.colorTokens(colorCount, colorsByNode);
    }

    String canonicalKey() {
        return Arrays.toString(colorsByNode);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof GraphColoringSolution solution
                && colorCount == solution.colorCount
                && Arrays.equals(colorsByNode, solution.colorsByNode);
    }

    @Override
    public int hashCode() {
        return 31 * Integer.hashCode(colorCount) + Arrays.hashCode(colorsByNode);
    }
}
