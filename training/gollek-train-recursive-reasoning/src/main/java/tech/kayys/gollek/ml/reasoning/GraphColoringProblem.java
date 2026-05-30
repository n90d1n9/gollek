package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Graph coloring input with optional fixed colors per node.
 */
public record GraphColoringProblem(
        int nodeCount,
        int colorCount,
        List<GraphColoringEdge> edges,
        int[] fixedColorsByNode) {
    public static final int UNCOLORED = -1;
    public static final int UNCOLORED_TOKEN = 0;

    public GraphColoringProblem {
        if (nodeCount < 1) {
            throw new IllegalArgumentException("nodeCount must be >= 1 but was " + nodeCount);
        }
        if (colorCount < 1) {
            throw new IllegalArgumentException("colorCount must be >= 1 but was " + colorCount);
        }
        edges = List.copyOf(edges == null ? List.of() : edges);
        requireEdges(nodeCount, edges);
        fixedColorsByNode = requireColors(nodeCount, colorCount, fixedColorsByNode, "fixedColorsByNode");
        requireFixedColorsDoNotConflict(edges, fixedColorsByNode);
    }

    public static GraphColoringProblem ofEdges(
            int nodeCount,
            int colorCount,
            GraphColoringEdge... edges) {
        return empty(nodeCount, colorCount, List.of(edges));
    }

    public static GraphColoringProblem empty(
            int nodeCount,
            int colorCount,
            List<GraphColoringEdge> edges) {
        int[] colors = new int[nodeCount];
        Arrays.fill(colors, UNCOLORED);
        return new GraphColoringProblem(nodeCount, colorCount, edges, colors);
    }

    public static GraphColoringProblem ofFixedColors(
            int colorCount,
            List<GraphColoringEdge> edges,
            int... fixedColorsByNode) {
        return new GraphColoringProblem(fixedColorsByNode.length, colorCount, edges, fixedColorsByNode);
    }

    @Override
    public int[] fixedColorsByNode() {
        return fixedColorsByNode.clone();
    }

    public boolean hasFixedColor(int node) {
        return fixedColorsByNode[node] != UNCOLORED;
    }

    public int fixedColor(int node) {
        return fixedColorsByNode[node];
    }

    public int[] toTokens() {
        return colorTokens(colorCount, fixedColorsByNode);
    }

    static int[] requireColors(
            int nodeCount,
            int colorCount,
            int[] colors,
            String name) {
        if (colors == null) {
            throw new NullPointerException(name + " must not be null");
        }
        if (colors.length != nodeCount) {
            throw new IllegalArgumentException(name + " length must be " + nodeCount + " but was " + colors.length);
        }
        int[] copy = colors.clone();
        for (int node = 0; node < copy.length; node++) {
            int color = copy[node];
            if (color < UNCOLORED || color >= colorCount) {
                throw new IllegalArgumentException(
                        name + "[" + node + "] must be -1 or in [0, " + (colorCount - 1) + "] but was " + color);
            }
        }
        return copy;
    }

    static int[] colorTokens(int colorCount, int[] colorsByNode) {
        int[] tokens = new int[colorsByNode.length];
        for (int node = 0; node < colorsByNode.length; node++) {
            int color = colorsByNode[node];
            tokens[node] = color == UNCOLORED ? UNCOLORED_TOKEN : color + 1;
        }
        return tokens;
    }

    private static void requireEdges(int nodeCount, List<GraphColoringEdge> edges) {
        Set<GraphColoringEdge> seen = new HashSet<>();
        for (GraphColoringEdge edge : edges) {
            if (edge.leftNode() >= nodeCount || edge.rightNode() >= nodeCount) {
                throw new IllegalArgumentException(
                        "edge " + edge + " references node outside [0, " + (nodeCount - 1) + "]");
            }
            if (!seen.add(edge)) {
                throw new IllegalArgumentException("duplicate edge: " + edge);
            }
        }
    }

    private static void requireFixedColorsDoNotConflict(
            List<GraphColoringEdge> edges,
            int[] fixedColorsByNode) {
        for (GraphColoringEdge edge : edges) {
            int left = fixedColorsByNode[edge.leftNode()];
            int right = fixedColorsByNode[edge.rightNode()];
            if (left != UNCOLORED && left == right) {
                throw new IllegalArgumentException("fixed colors conflict on edge " + edge);
            }
        }
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof GraphColoringProblem problem
                && nodeCount == problem.nodeCount
                && colorCount == problem.colorCount
                && edges.equals(problem.edges)
                && Arrays.equals(fixedColorsByNode, problem.fixedColorsByNode);
    }

    @Override
    public int hashCode() {
        int result = 31 * Integer.hashCode(nodeCount) + Integer.hashCode(colorCount);
        result = 31 * result + edges.hashCode();
        return 31 * result + Arrays.hashCode(fixedColorsByNode);
    }
}
