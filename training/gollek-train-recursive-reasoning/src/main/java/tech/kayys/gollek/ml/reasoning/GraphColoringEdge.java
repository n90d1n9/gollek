package tech.kayys.gollek.ml.reasoning;

/**
 * Undirected edge between two graph-coloring nodes.
 */
public record GraphColoringEdge(int leftNode, int rightNode) {
    public GraphColoringEdge {
        if (leftNode < 0) {
            throw new IllegalArgumentException("leftNode must be >= 0 but was " + leftNode);
        }
        if (rightNode < 0) {
            throw new IllegalArgumentException("rightNode must be >= 0 but was " + rightNode);
        }
        if (leftNode == rightNode) {
            throw new IllegalArgumentException("self edges are not allowed: " + leftNode);
        }
        if (leftNode > rightNode) {
            int oldLeft = leftNode;
            leftNode = rightNode;
            rightNode = oldLeft;
        }
    }

    public static GraphColoringEdge of(int leftNode, int rightNode) {
        return new GraphColoringEdge(leftNode, rightNode);
    }
}
