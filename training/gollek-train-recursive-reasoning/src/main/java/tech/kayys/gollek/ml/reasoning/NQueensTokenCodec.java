package tech.kayys.gollek.ml.reasoning;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * Token codec for N-Queens boards using the paper convention: 0=pad, 1=empty, 2=queen.
 */
public final class NQueensTokenCodec {
    public static final int PAD_TOKEN = 0;

    private NQueensTokenCodec() {
    }

    public static NQueensTokenDecodeResult decodeSolution(int size, int[] tokens) {
        requireSize(size);
        int[] input = requireTokens(size, tokens);
        int[] columnsByRow = new int[size];
        Arrays.fill(columnsByRow, NQueensProblem.EMPTY);

        int invalidTokenCount = 0;
        int emptyRowCount = 0;
        int multiQueenRowCount = 0;
        int queenTokenCount = 0;

        for (int row = 0; row < size; row++) {
            int firstQueenColumn = NQueensProblem.EMPTY;
            int queensInRow = 0;
            for (int column = 0; column < size; column++) {
                int token = input[row * size + column];
                if (token == NQueensProblem.QUEEN_TOKEN) {
                    queenTokenCount++;
                    queensInRow++;
                    if (firstQueenColumn == NQueensProblem.EMPTY) {
                        firstQueenColumn = column;
                    }
                } else if (token != PAD_TOKEN && token != NQueensProblem.EMPTY_TOKEN) {
                    invalidTokenCount++;
                }
            }
            if (queensInRow == 1) {
                columnsByRow[row] = firstQueenColumn;
            } else if (queensInRow == 0) {
                emptyRowCount++;
            } else {
                multiQueenRowCount++;
            }
        }

        return new NQueensTokenDecodeResult(
                new NQueensSolution(columnsByRow),
                invalidTokenCount,
                emptyRowCount,
                multiQueenRowCount,
                queenTokenCount,
                Map.of("size", size));
    }

    public static NQueensProblem decodeProblem(int size, int[] tokens) {
        NQueensTokenDecodeResult decoded = decodeSolution(size, tokens);
        if (decoded.invalidTokenCount() > 0 || decoded.multiQueenRowCount() > 0) {
            throw new IllegalArgumentException(
                    "problem tokens must not contain invalid tokens or multiple queens in a row");
        }
        return new NQueensProblem(size, decoded.solution().columnsByRow());
    }

    public static int[] encode(NQueensProblem problem) {
        return Objects.requireNonNull(problem, "problem must not be null").toTokens();
    }

    public static int[] encode(NQueensSolution solution) {
        return Objects.requireNonNull(solution, "solution must not be null").toTokens();
    }

    private static int[] requireTokens(int size, int[] tokens) {
        Objects.requireNonNull(tokens, "tokens must not be null");
        int expectedLength = size * size;
        if (tokens.length != expectedLength) {
            throw new IllegalArgumentException(
                    "tokens length must be " + expectedLength + " for board size " + size + " but was " + tokens.length);
        }
        return tokens.clone();
    }

    private static void requireSize(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size must be >= 1 but was " + size);
        }
    }
}
