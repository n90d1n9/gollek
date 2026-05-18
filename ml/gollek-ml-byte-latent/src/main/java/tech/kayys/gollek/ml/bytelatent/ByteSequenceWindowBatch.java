package tech.kayys.gollek.ml.bytelatent;

import java.util.Arrays;

/**
 * Packed byte-level causal language-model batch with padding and attention mask.
 */
public record ByteSequenceWindowBatch(
        int[][] inputIds,
        int[][] targetIds,
        boolean[][] attentionMask,
        int padTokenId,
        int sequenceLength) {

    public ByteSequenceWindowBatch {
        inputIds = deepCopy(inputIds, "inputIds");
        targetIds = deepCopy(targetIds, "targetIds");
        attentionMask = deepCopy(attentionMask, "attentionMask");
        if (padTokenId < 0) {
            throw new IllegalArgumentException("padTokenId must be >= 0 but was " + padTokenId);
        }
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("sequenceLength must be > 0 but was " + sequenceLength);
        }
        validateSameShape(inputIds, targetIds, attentionMask, sequenceLength);
    }

    @Override
    public int[][] inputIds() {
        return deepCopy(inputIds, "inputIds");
    }

    @Override
    public int[][] targetIds() {
        return deepCopy(targetIds, "targetIds");
    }

    @Override
    public boolean[][] attentionMask() {
        return deepCopy(attentionMask, "attentionMask");
    }

    public int batchSize() {
        return inputIds.length;
    }

    private static int[][] deepCopy(int[][] values, String label) {
        if (values == null) {
            throw new IllegalArgumentException(label + " must not be null");
        }
        int[][] copy = new int[values.length][];
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null) {
                throw new IllegalArgumentException(label + "[" + i + "] must not be null");
            }
            copy[i] = Arrays.copyOf(values[i], values[i].length);
        }
        return copy;
    }

    private static boolean[][] deepCopy(boolean[][] values, String label) {
        if (values == null) {
            throw new IllegalArgumentException(label + " must not be null");
        }
        boolean[][] copy = new boolean[values.length][];
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null) {
                throw new IllegalArgumentException(label + "[" + i + "] must not be null");
            }
            copy[i] = Arrays.copyOf(values[i], values[i].length);
        }
        return copy;
    }

    private static void validateSameShape(
            int[][] inputIds,
            int[][] targetIds,
            boolean[][] attentionMask,
            int sequenceLength) {
        if (inputIds.length != targetIds.length || inputIds.length != attentionMask.length) {
            throw new IllegalArgumentException("inputIds, targetIds, and attentionMask must have the same batch size");
        }
        for (int i = 0; i < inputIds.length; i++) {
            if (inputIds[i].length != sequenceLength) {
                throw new IllegalArgumentException(
                        "inputIds[" + i + "] length must equal sequenceLength=" + sequenceLength
                                + " but was " + inputIds[i].length);
            }
            if (targetIds[i].length != sequenceLength) {
                throw new IllegalArgumentException(
                        "targetIds[" + i + "] length must equal sequenceLength=" + sequenceLength
                                + " but was " + targetIds[i].length);
            }
            if (attentionMask[i].length != sequenceLength) {
                throw new IllegalArgumentException(
                        "attentionMask[" + i + "] length must equal sequenceLength=" + sequenceLength
                                + " but was " + attentionMask[i].length);
            }
        }
    }
}
