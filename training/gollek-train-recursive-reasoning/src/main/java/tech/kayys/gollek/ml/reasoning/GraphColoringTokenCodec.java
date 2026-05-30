package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * Token codec for graph coloring: 0=uncolored, 1..colorCount=color ids + 1.
 */
public final class GraphColoringTokenCodec {
    private GraphColoringTokenCodec() {
    }

    public static GraphColoringTokenDecodeResult decodeSolution(
            int nodeCount,
            int colorCount,
            int[] tokens) {
        requireShape(nodeCount, colorCount, tokens);
        int[] colorsByNode = new int[nodeCount];
        Arrays.fill(colorsByNode, GraphColoringProblem.UNCOLORED);

        int invalidTokenCount = 0;
        int uncoloredNodeCount = 0;
        int colorTokenCount = 0;
        for (int node = 0; node < nodeCount; node++) {
            int token = tokens[node];
            if (token == GraphColoringProblem.UNCOLORED_TOKEN) {
                uncoloredNodeCount++;
            } else if (token >= 1 && token <= colorCount) {
                colorsByNode[node] = token - 1;
                colorTokenCount++;
            } else {
                invalidTokenCount++;
                uncoloredNodeCount++;
            }
        }

        return new GraphColoringTokenDecodeResult(
                new GraphColoringSolution(colorCount, colorsByNode),
                invalidTokenCount,
                uncoloredNodeCount,
                colorTokenCount,
                Map.of(
                        "nodeCount", nodeCount,
                        "colorCount", colorCount));
    }

    public static GraphColoringProblem decodeProblem(
            int colorCount,
            java.util.List<GraphColoringEdge> edges,
            int[] tokens) {
        Objects.requireNonNull(tokens, "tokens must not be null");
        GraphColoringTokenDecodeResult decoded = decodeSolution(tokens.length, colorCount, tokens);
        if (decoded.invalidTokenCount() > 0) {
            throw new IllegalArgumentException("problem tokens must not contain invalid color tokens");
        }
        return new GraphColoringProblem(tokens.length, colorCount, edges, decoded.solution().colorsByNode());
    }

    public static int[] encode(GraphColoringProblem problem) {
        return Objects.requireNonNull(problem, "problem must not be null").toTokens();
    }

    public static int[] encode(GraphColoringSolution solution) {
        return Objects.requireNonNull(solution, "solution must not be null").toTokens();
    }

    private static void requireShape(int nodeCount, int colorCount, int[] tokens) {
        if (nodeCount < 1) {
            throw new IllegalArgumentException("nodeCount must be >= 1 but was " + nodeCount);
        }
        if (colorCount < 1) {
            throw new IllegalArgumentException("colorCount must be >= 1 but was " + colorCount);
        }
        Objects.requireNonNull(tokens, "tokens must not be null");
        if (tokens.length != nodeCount) {
            throw new IllegalArgumentException(
                    "tokens length must be " + nodeCount + " but was " + tokens.length);
        }
    }
}
