package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Converts model logits into N-Queens board tokens without depending on a
 * specific tensor backend.
 */
public final class NQueensTokenProjector {
    private NQueensTokenProjector() {
    }

    public static NQueensTokenProjectionResult cellArgmax(int size, int vocabSize, float[] logits) {
        requireBoard(size);
        requireBoardVocabulary(vocabSize);
        DiscreteTokenProjectionResult projection = DiscreteTokenProjector.argmax(size * size, vocabSize, logits);
        return new NQueensTokenProjectionResult(
                projection.tokens(),
                NQueensTokenProjectionMode.CELL_ARGMAX,
                metadata(size, vocabSize, 0));
    }

    public static NQueensTokenProjectionResult rowQueenArgmax(
            NQueensProblem problem,
            int vocabSize,
            float[] logits) {
        return rowQueenArgmax(problem, vocabSize, logits, false);
    }

    public static NQueensTokenProjectionResult rowQueenArgmaxRespectFixed(
            NQueensProblem problem,
            int vocabSize,
            float[] logits) {
        return rowQueenArgmax(problem, vocabSize, logits, true);
    }

    private static NQueensTokenProjectionResult rowQueenArgmax(
            NQueensProblem problem,
            int vocabSize,
            float[] logits,
            boolean respectFixedQueens) {
        Objects.requireNonNull(problem, "problem must not be null");
        requireBoardVocabulary(vocabSize);
        float[] input = DiscreteTokenProjector.requireFiniteLogits(
                problem.size() * problem.size(),
                vocabSize,
                logits);
        int[] tokens = new int[problem.size() * problem.size()];
        Arrays.fill(tokens, NQueensProblem.EMPTY_TOKEN);

        int forcedFixedQueenCount = 0;
        for (int row = 0; row < problem.size(); row++) {
            int column = respectFixedQueens && problem.hasFixedQueen(row)
                    ? problem.fixedColumn(row)
                    : bestQueenColumn(problem.size(), vocabSize, input, row);
            if (respectFixedQueens && problem.hasFixedQueen(row)) {
                forcedFixedQueenCount++;
            }
            tokens[row * problem.size() + column] = NQueensProblem.QUEEN_TOKEN;
        }

        NQueensTokenProjectionMode mode = respectFixedQueens
                ? NQueensTokenProjectionMode.ROW_QUEEN_ARGMAX_RESPECT_FIXED
                : NQueensTokenProjectionMode.ROW_QUEEN_ARGMAX;
        return new NQueensTokenProjectionResult(
                tokens,
                mode,
                metadata(problem.size(), vocabSize, forcedFixedQueenCount));
    }

    private static int bestQueenColumn(int size, int vocabSize, float[] logits, int row) {
        int bestCell = DiscreteTokenProjector.bestItemForToken(
                logits,
                row * size,
                size,
                vocabSize,
                NQueensProblem.QUEEN_TOKEN);
        return bestCell - row * size;
    }

    private static void requireBoardVocabulary(int vocabSize) {
        if (vocabSize <= NQueensProblem.QUEEN_TOKEN) {
            throw new IllegalArgumentException(
                    "vocabSize must include pad, empty, and queen tokens but was " + vocabSize);
        }
    }

    private static void requireBoard(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size must be >= 1 but was " + size);
        }
    }

    private static Map<String, Object> metadata(int size, int vocabSize, int forcedFixedQueenCount) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("size", size);
        metadata.put("vocabSize", vocabSize);
        metadata.put("cellCount", size * size);
        metadata.put("forcedFixedQueenCount", forcedFixedQueenCount);
        return metadata;
    }
}
