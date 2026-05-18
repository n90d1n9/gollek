package tech.kayys.gollek.ml.bytelatent;

import java.util.List;
import java.util.Objects;

/**
 * Utilities for packing raw byte sequences into causal language-model windows.
 */
public final class ByteSequenceCollator {
    private ByteSequenceCollator() {
    }

    public static ByteSequenceWindowBatch causalLanguageModeling(
            ByteSequenceBatch batch,
            int windowLength,
            int padTokenId) {
        Objects.requireNonNull(batch, "batch must not be null");
        if (windowLength <= 0) {
            throw new IllegalArgumentException("windowLength must be > 0 but was " + windowLength);
        }
        if (padTokenId < 0) {
            throw new IllegalArgumentException("padTokenId must be >= 0 but was " + padTokenId);
        }
        List<byte[]> sequences = batch.sequences();
        int[][] inputIds = new int[sequences.size()][windowLength];
        int[][] targetIds = new int[sequences.size()][windowLength];
        boolean[][] attentionMask = new boolean[sequences.size()][windowLength];

        for (int row = 0; row < sequences.size(); row++) {
            fillPadding(inputIds[row], padTokenId);
            fillPadding(targetIds[row], padTokenId);

            byte[] sequence = sequences.get(row);
            int availableTokens = Math.max(0, sequence.length - 1);
            int copied = Math.min(windowLength, availableTokens);
            for (int i = 0; i < copied; i++) {
                inputIds[row][i] = Byte.toUnsignedInt(sequence[i]);
                targetIds[row][i] = Byte.toUnsignedInt(sequence[i + 1]);
                attentionMask[row][i] = true;
            }
        }

        return new ByteSequenceWindowBatch(inputIds, targetIds, attentionMask, padTokenId, windowLength);
    }

    private static void fillPadding(int[] row, int padTokenId) {
        for (int i = 0; i < row.length; i++) {
            row[i] = padTokenId;
        }
    }
}
