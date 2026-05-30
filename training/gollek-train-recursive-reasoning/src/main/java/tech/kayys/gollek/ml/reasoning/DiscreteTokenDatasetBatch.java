package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Backend-neutral padded token batch for structured reasoning trainers.
 */
public record DiscreteTokenDatasetBatch(
        String[] taskIds,
        int[] exampleIndices,
        int[][] inputTokens,
        int[][] targetTokens,
        int[][] inputMask,
        int[][] targetMask,
        int[] inputLengths,
        int[] targetLengths,
        int[] knownSolutionCounts,
        List<Map<String, Object>> metadata,
        int inputPadToken,
        int targetPadToken) {

    public DiscreteTokenDatasetBatch {
        taskIds = Objects.requireNonNull(taskIds, "taskIds must not be null").clone();
        if (taskIds.length == 0) {
            throw new IllegalArgumentException("batch must contain at least one example");
        }
        int batchSize = taskIds.length;
        for (int row = 0; row < batchSize; row++) {
            String taskId = Objects.requireNonNull(taskIds[row], "taskIds[" + row + "] must not be null");
            if (taskId.isBlank()) {
                throw new IllegalArgumentException("taskIds[" + row + "] must not be blank");
            }
        }

        exampleIndices = copyVector(exampleIndices, batchSize, "exampleIndices");
        inputTokens = copyMatrix(inputTokens, batchSize, "inputTokens");
        targetTokens = copyMatrix(targetTokens, batchSize, "targetTokens");
        inputMask = copyMatrix(inputMask, batchSize, "inputMask");
        targetMask = copyMatrix(targetMask, batchSize, "targetMask");
        inputLengths = copyVector(inputLengths, batchSize, "inputLengths");
        targetLengths = copyVector(targetLengths, batchSize, "targetLengths");
        knownSolutionCounts = copyVector(knownSolutionCounts, batchSize, "knownSolutionCounts");
        metadata = copyMetadata(metadata, batchSize);

        int maxInputLength = columnCount(inputTokens, "inputTokens");
        int maxTargetLength = columnCount(targetTokens, "targetTokens");
        requireColumnCount(inputMask, maxInputLength, "inputMask");
        requireColumnCount(targetMask, maxTargetLength, "targetMask");

        if (maxInputLength == 0) {
            throw new IllegalArgumentException("inputTokens must have at least one column");
        }
        if (maxTargetLength == 0) {
            throw new IllegalArgumentException("targetTokens must have at least one column");
        }

        for (int row = 0; row < batchSize; row++) {
            if (exampleIndices[row] < 0) {
                throw new IllegalArgumentException("exampleIndices[" + row + "] must be >= 0");
            }
            validateLength(inputLengths[row], maxInputLength, "inputLengths[" + row + "]");
            validateLength(targetLengths[row], maxTargetLength, "targetLengths[" + row + "]");
            validateKnownSolutionCount(knownSolutionCounts[row], row);
            validateMaskRow(inputMask[row], inputLengths[row], "inputMask[" + row + "]");
            validateMaskRow(targetMask[row], targetLengths[row], "targetMask[" + row + "]");
        }
    }

    public int batchSize() {
        return taskIds.length;
    }

    public int maxInputLength() {
        return inputTokens[0].length;
    }

    public int maxTargetLength() {
        return targetTokens[0].length;
    }

    public int inputTokenCount(int row) {
        return inputLengths[checkedRow(row)];
    }

    public int targetTokenCount(int row) {
        return targetLengths[checkedRow(row)];
    }

    public boolean hasKnownSolutionCount(int row) {
        return knownSolutionCounts[checkedRow(row)] > 0;
    }

    @Override
    public String[] taskIds() {
        return taskIds.clone();
    }

    @Override
    public int[] exampleIndices() {
        return exampleIndices.clone();
    }

    @Override
    public int[][] inputTokens() {
        return copyMatrix(inputTokens);
    }

    @Override
    public int[][] targetTokens() {
        return copyMatrix(targetTokens);
    }

    @Override
    public int[][] inputMask() {
        return copyMatrix(inputMask);
    }

    @Override
    public int[][] targetMask() {
        return copyMatrix(targetMask);
    }

    @Override
    public int[] inputLengths() {
        return inputLengths.clone();
    }

    @Override
    public int[] targetLengths() {
        return targetLengths.clone();
    }

    @Override
    public int[] knownSolutionCounts() {
        return knownSolutionCounts.clone();
    }

    private int checkedRow(int row) {
        if (row < 0 || row >= batchSize()) {
            throw new IndexOutOfBoundsException("row out of range: " + row);
        }
        return row;
    }

    private static int[] copyVector(int[] values, int expectedLength, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.length != expectedLength) {
            throw new IllegalArgumentException(
                    name + " length must be " + expectedLength + " but was " + values.length);
        }
        return values.clone();
    }

    private static int[][] copyMatrix(int[][] values, int expectedRows, String name) {
        Objects.requireNonNull(values, name + " must not be null");
        if (values.length != expectedRows) {
            throw new IllegalArgumentException(
                    name + " row count must be " + expectedRows + " but was " + values.length);
        }
        return copyMatrix(values);
    }

    private static int[][] copyMatrix(int[][] values) {
        int[][] copy = new int[values.length][];
        for (int row = 0; row < values.length; row++) {
            copy[row] = Objects.requireNonNull(values[row], "matrix row " + row + " must not be null").clone();
        }
        return copy;
    }

    private static int columnCount(int[][] values, String name) {
        int columns = values[0].length;
        requireColumnCount(values, columns, name);
        return columns;
    }

    private static void requireColumnCount(int[][] values, int expectedColumns, String name) {
        for (int row = 0; row < values.length; row++) {
            if (values[row].length != expectedColumns) {
                throw new IllegalArgumentException(
                        name + " must be rectangular; row " + row + " had " + values[row].length
                                + " columns instead of " + expectedColumns);
            }
        }
    }

    private static void validateLength(int length, int maxLength, String name) {
        if (length <= 0 || length > maxLength) {
            throw new IllegalArgumentException(name + " must be between 1 and " + maxLength + " but was " + length);
        }
    }

    private static void validateKnownSolutionCount(int knownSolutionCount, int row) {
        if (knownSolutionCount < -1 || knownSolutionCount == 0) {
            throw new IllegalArgumentException(
                    "knownSolutionCounts[" + row + "] must be -1 or >= 1 but was " + knownSolutionCount);
        }
    }

    private static void validateMaskRow(int[] mask, int length, String name) {
        for (int column = 0; column < mask.length; column++) {
            int expected = column < length ? 1 : 0;
            if (mask[column] != expected) {
                throw new IllegalArgumentException(
                        name + "[" + column + "] must be " + expected + " for sequence length " + length);
            }
        }
    }

    private static List<Map<String, Object>> copyMetadata(List<Map<String, Object>> metadata, int expectedLength) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        if (metadata.size() != expectedLength) {
            throw new IllegalArgumentException(
                    "metadata size must be " + expectedLength + " but was " + metadata.size());
        }
        List<Map<String, Object>> copy = new ArrayList<>(metadata.size());
        for (int row = 0; row < metadata.size(); row++) {
            Map<String, Object> rowMetadata = metadata.get(row);
            copy.add(rowMetadata == null ? Map.of() : Map.copyOf(rowMetadata));
        }
        return List.copyOf(copy);
    }
}
