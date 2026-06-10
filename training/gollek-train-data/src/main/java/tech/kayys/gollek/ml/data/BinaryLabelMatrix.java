package tech.kayys.gollek.ml.data;

import java.util.Objects;

record BinaryLabelMatrix(float[] values, int rows, int columns) {

    static BinaryLabelMatrix from(int[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            int[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                int value = labelRow[column];
                if (value != 0 && value != 1) {
                    throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + value);
                }
                values[row * columns + column] = value;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    static BinaryLabelMatrix from(boolean[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            boolean[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                values[row * columns + column] = labelRow[column] ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    static BinaryLabelMatrix from(float[][] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int columns = matrixColumns(labels.length, labels.length == 0 ? 0 : rowLength(labels[0], 0));
        float[] values = new float[labels.length * columns];
        for (int row = 0; row < labels.length; row++) {
            float[] labelRow = Objects.requireNonNull(labels[row], "label row must not be null");
            requireRowWidth(row, labelRow.length, columns);
            for (int column = 0; column < columns; column++) {
                float value = labelRow[column];
                if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                    throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
                }
                values[row * columns + column] = value >= 0.5f ? 1f : 0f;
            }
        }
        return new BinaryLabelMatrix(values, labels.length, columns);
    }

    private static int matrixColumns(int rows, int columns) {
        if (rows == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one row");
        }
        if (columns == 0) {
            throw new IllegalArgumentException("binary label matrix must contain at least one column");
        }
        return columns;
    }

    private static int rowLength(int[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(boolean[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static int rowLength(float[] row, int rowIndex) {
        return Objects.requireNonNull(row, "label row " + rowIndex + " must not be null").length;
    }

    private static void requireRowWidth(int row, int actual, int expected) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "binary label matrix must be rectangular; row "
                            + row + " has " + actual + " columns, expected " + expected);
        }
    }
}
