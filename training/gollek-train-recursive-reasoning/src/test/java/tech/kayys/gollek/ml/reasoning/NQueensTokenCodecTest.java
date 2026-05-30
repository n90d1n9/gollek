package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class NQueensTokenCodecTest {
    @Test
    void decodesCleanSolutionTokens() {
        int[] tokens = NQueensSolution.ofColumns(1, 3, 0, 2).toTokens();

        NQueensTokenDecodeResult decoded = NQueensTokenCodec.decodeSolution(4, tokens);

        assertTrue(decoded.clean());
        assertEquals(NQueensSolution.ofColumns(1, 3, 0, 2), decoded.solution());
        assertEquals(4, decoded.queenTokenCount());
        assertEquals(0, decoded.invalidTokenCount());
        assertEquals(0, decoded.multiQueenRowCount());
    }

    @Test
    void decodesMalformedTokensWithDiagnostics() {
        int[] tokens = {
            1, 2, 1, 1,
            1, 1, 1, 1,
            2, 1, 2, 1,
            9, 1, 1, 1
        };

        NQueensTokenDecodeResult decoded = NQueensTokenCodec.decodeSolution(4, tokens);

        assertFalse(decoded.clean());
        assertEquals(NQueensSolution.ofColumns(1, -1, -1, -1), decoded.solution());
        assertEquals(1, decoded.invalidTokenCount());
        assertEquals(2, decoded.emptyRowCount());
        assertEquals(1, decoded.multiQueenRowCount());
        assertEquals(3, decoded.queenTokenCount());
    }

    @Test
    void evaluatesTokenPredictionsWithDecodeDiagnostics() {
        int[] tokens = {
            1, 2, 1, 1,
            1, 1, 1, 1,
            2, 1, 2, 1,
            9, 1, 1, 1
        };

        NQueensEvaluation evaluation = NQueensBenchmark.evaluateTokens(NQueensProblem.empty(4), tokens);

        assertFalse(evaluation.valid());
        assertEquals(3, evaluation.missingRowCount());
        assertEquals(5, evaluation.conflictCount());
        assertEquals(1, evaluation.metadata().get("invalidTokenCount"));
        assertEquals(1, evaluation.metadata().get("multiQueenRowCount"));
    }

    @Test
    void decodesPartialProblemTokensAndTreatsPadAsEmpty() {
        int[] tokens = {
            0, 0, 0, 0,
            0, 0, 2, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        };

        NQueensProblem problem = NQueensTokenCodec.decodeProblem(4, tokens);

        assertEquals(NQueensProblem.ofFixedColumns(-1, 2, -1, -1), problem);
        assertArrayEquals(NQueensProblem.ofFixedColumns(-1, 2, -1, -1).toTokens(), NQueensTokenCodec.encode(problem));
    }

    @Test
    void coverageCanConsumePredictedTokenBoards() {
        NQueensCoverageReport report = NQueensBenchmark.coverageTokens(
                NQueensProblem.empty(4),
                List.of(
                        NQueensSolution.ofColumns(1, 3, 0, 2).toTokens(),
                        NQueensSolution.ofColumns(1, 3, 0, 2).toTokens(),
                        new int[] {
                            1, 2, 1, 1,
                            1, 1, 1, 1,
                            2, 1, 2, 1,
                            9, 1, 1, 1
                        }),
                2);

        assertEquals(3, report.candidateCount());
        assertEquals(2, report.validCandidateCount());
        assertEquals(1, report.uniqueValidSolutionCount());
        assertEquals(1, report.duplicateValidCandidateCount());
        assertEquals(0.5, report.coverageRate(), 1e-9);
    }

    @Test
    void rejectsInvalidTokenShapesAndAmbiguousProblemTokens() {
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenCodec.decodeSolution(0, new int[] {}));
        assertThrows(IllegalArgumentException.class, () -> NQueensTokenCodec.decodeSolution(4, new int[] {1, 2}));
        assertThrows(
                IllegalArgumentException.class,
                () -> NQueensTokenCodec.decodeProblem(
                        2,
                        new int[] {
                            2, 2,
                            1, 1
                        }));
    }
}
