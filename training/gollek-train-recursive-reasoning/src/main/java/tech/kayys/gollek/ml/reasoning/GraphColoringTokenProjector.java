package tech.kayys.gollek.ml.reasoning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Converts model logits into graph-coloring tokens without depending on a
 * specific tensor backend.
 */
public final class GraphColoringTokenProjector {
    private GraphColoringTokenProjector() {
    }

    public static GraphColoringTokenProjectionResult nodeArgmax(
            GraphColoringProblem problem,
            int vocabSize,
            float[] logits) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireColorVocabulary(problem, vocabSize);
        DiscreteTokenProjectionResult projection =
                DiscreteTokenProjector.argmax(problem.nodeCount(), vocabSize, logits);
        return new GraphColoringTokenProjectionResult(
                projection.tokens(),
                GraphColoringTokenProjectionMode.NODE_ARGMAX,
                metadata(problem, vocabSize, 0));
    }

    public static GraphColoringTokenProjectionResult nodeColorArgmax(
            GraphColoringProblem problem,
            int vocabSize,
            float[] logits) {
        return nodeColorArgmax(problem, vocabSize, logits, false);
    }

    public static GraphColoringTokenProjectionResult nodeColorArgmaxRespectFixed(
            GraphColoringProblem problem,
            int vocabSize,
            float[] logits) {
        return nodeColorArgmax(problem, vocabSize, logits, true);
    }

    private static GraphColoringTokenProjectionResult nodeColorArgmax(
            GraphColoringProblem problem,
            int vocabSize,
            float[] logits,
            boolean respectFixedColors) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireColorVocabulary(problem, vocabSize);
        float[] input = DiscreteTokenProjector.requireFiniteLogits(problem.nodeCount(), vocabSize, logits);

        int[] tokens = new int[problem.nodeCount()];
        int forcedFixedColorCount = 0;
        for (int node = 0; node < problem.nodeCount(); node++) {
            if (respectFixedColors && problem.hasFixedColor(node)) {
                tokens[node] = problem.fixedColor(node) + 1;
                forcedFixedColorCount++;
            } else {
                tokens[node] = bestColorToken(input, node, vocabSize, problem.colorCount());
            }
        }

        GraphColoringTokenProjectionMode mode = respectFixedColors
                ? GraphColoringTokenProjectionMode.NODE_COLOR_ARGMAX_RESPECT_FIXED
                : GraphColoringTokenProjectionMode.NODE_COLOR_ARGMAX;
        return new GraphColoringTokenProjectionResult(
                tokens,
                mode,
                metadata(problem, vocabSize, forcedFixedColorCount));
    }

    private static int bestColorToken(float[] logits, int node, int vocabSize, int colorCount) {
        int offset = node * vocabSize;
        int bestToken = 1;
        float bestScore = logits[offset + bestToken];
        for (int token = 2; token <= colorCount; token++) {
            float score = logits[offset + token];
            if (score > bestScore) {
                bestScore = score;
                bestToken = token;
            }
        }
        return bestToken;
    }

    private static void requireColorVocabulary(GraphColoringProblem problem, int vocabSize) {
        if (vocabSize <= problem.colorCount()) {
            throw new IllegalArgumentException(
                    "vocabSize must include uncolored and one token per color but was "
                            + vocabSize + " for colorCount " + problem.colorCount());
        }
    }

    private static Map<String, Object> metadata(
            GraphColoringProblem problem,
            int vocabSize,
            int forcedFixedColorCount) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("nodeCount", problem.nodeCount());
        metadata.put("colorCount", problem.colorCount());
        metadata.put("edgeCount", problem.edges().size());
        metadata.put("vocabSize", vocabSize);
        metadata.put("forcedFixedColorCount", forcedFixedColorCount);
        return metadata;
    }
}
