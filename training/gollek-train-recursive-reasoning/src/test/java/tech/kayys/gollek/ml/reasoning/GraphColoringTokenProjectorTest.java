package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class GraphColoringTokenProjectorTest {
    @Test
    void nodeArgmaxProjectsPerNodeTokenLogits() {
        GraphColoringProblem problem = triangle();
        int[] expectedTokens = GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens();

        GraphColoringTokenProjectionResult result = GraphColoringTokenProjector.nodeArgmax(
                problem,
                4,
                logitsForTokens(expectedTokens, 4));

        assertArrayEquals(expectedTokens, result.tokens());
        assertEquals(GraphColoringTokenProjectionMode.NODE_ARGMAX, result.mode());
        assertEquals(3, result.metadata().get("nodeCount"));
        result.tokens()[0] = 9;
        assertArrayEquals(expectedTokens, result.tokens());
    }

    @Test
    void nodeArgmaxCanExposeInvalidTokensForDiagnostics() {
        GraphColoringProblem problem = triangle();
        int[] rawTokens = {1, 4, 3};

        GraphColoringTokenProjectionResult result = GraphColoringTokenProjector.nodeArgmax(
                problem,
                5,
                logitsForTokens(rawTokens, 5));

        assertArrayEquals(rawTokens, result.tokens());
        assertFalse(GraphColoringBenchmark.evaluateTokens(problem, result.tokens()).valid());
        assertEquals(1, GraphColoringBenchmark.evaluateTokens(problem, result.tokens()).invalidTokenCount());
    }

    @Test
    void nodeColorArgmaxChoosesOnlyLegalColorTokens() {
        GraphColoringProblem problem = triangle();
        float[] logits = lowLogits(3, 5);
        setScore(logits, 5, 0, 0, 9.0f);
        setScore(logits, 5, 0, 4, 8.0f);
        setScore(logits, 5, 0, 1, 2.0f);
        setScore(logits, 5, 1, 2, 3.0f);
        setScore(logits, 5, 1, 4, 10.0f);
        setScore(logits, 5, 2, 3, 4.0f);

        GraphColoringTokenProjectionResult result = GraphColoringTokenProjector.nodeColorArgmax(
                problem,
                5,
                logits);

        assertArrayEquals(GraphColoringSolution.ofColors(3, 0, 1, 2).toTokens(), result.tokens());
        assertTrue(GraphColoringBenchmark.evaluateTokens(problem, result.tokens()).valid());
        assertEquals(GraphColoringTokenProjectionMode.NODE_COLOR_ARGMAX, result.mode());
    }

    @Test
    void nodeColorArgmaxCanRespectFixedColors() {
        GraphColoringProblem problem = GraphColoringProblem.ofFixedColors(
                3,
                List.of(GraphColoringEdge.of(0, 1), GraphColoringEdge.of(1, 2)),
                2,
                -1,
                -1);
        float[] logits = lowLogits(3, 4);
        setScore(logits, 4, 0, 1, 9.0f);
        setScore(logits, 4, 0, 3, -2.0f);
        setScore(logits, 4, 1, 1, 3.0f);
        setScore(logits, 4, 2, 2, 3.0f);

        GraphColoringTokenProjectionResult result = GraphColoringTokenProjector.nodeColorArgmaxRespectFixed(
                problem,
                4,
                logits);

        assertArrayEquals(GraphColoringSolution.ofColors(3, 2, 0, 1).toTokens(), result.tokens());
        assertEquals(GraphColoringTokenProjectionMode.NODE_COLOR_ARGMAX_RESPECT_FIXED, result.mode());
        assertEquals(1, result.metadata().get("forcedFixedColorCount"));
    }

    @Test
    void rejectsInvalidProjectionInputs() {
        GraphColoringProblem problem = triangle();

        assertThrows(IllegalArgumentException.class, () -> GraphColoringTokenProjector.nodeArgmax(
                problem,
                3,
                new float[9]));
        assertThrows(IllegalArgumentException.class, () -> GraphColoringTokenProjector.nodeArgmax(
                problem,
                4,
                new float[8]));

        float[] logits = new float[12];
        logits[2] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> GraphColoringTokenProjector.nodeColorArgmax(
                problem,
                4,
                logits));
    }

    private static float[] logitsForTokens(int[] tokens, int vocabSize) {
        float[] logits = lowLogits(tokens.length, vocabSize);
        for (int node = 0; node < tokens.length; node++) {
            logits[node * vocabSize + tokens[node]] = 4.0f;
        }
        return logits;
    }

    private static float[] lowLogits(int nodeCount, int vocabSize) {
        float[] logits = new float[nodeCount * vocabSize];
        Arrays.fill(logits, -4.0f);
        return logits;
    }

    private static void setScore(float[] logits, int vocabSize, int node, int token, float score) {
        logits[node * vocabSize + token] = score;
    }

    private static GraphColoringProblem triangle() {
        return GraphColoringProblem.ofEdges(
                3,
                3,
                GraphColoringEdge.of(0, 1),
                GraphColoringEdge.of(1, 2),
                GraphColoringEdge.of(0, 2));
    }
}
