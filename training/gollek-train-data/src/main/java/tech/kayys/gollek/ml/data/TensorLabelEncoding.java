package tech.kayys.gollek.ml.data;

import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;

final class TensorLabelEncoding {
    private TensorLabelEncoding() {
    }

    static DataLoader.TensorDataset tensorDataset(GradTensor inputs, GradTensor labels) {
        return new DataLoader.TensorDataset(inputs, labels);
    }

    static GradTensor classLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length);
    }

    static GradTensor binaryLabels(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] != 0 && labels[i] != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + labels[i]);
            }
            values[i] = labels[i];
        }
        return GradTensor.of(values, labels.length, 1);
    }

    static GradTensor binaryLabels(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            values[i] = labels[i] ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    static GradTensor binaryLabels(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        float[] values = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            float value = labels[i];
            if (Math.abs(value) > 1e-6f && Math.abs(value - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + value);
            }
            values[i] = value >= 0.5f ? 1f : 0f;
        }
        return GradTensor.of(values, labels.length, 1);
    }

    static GradTensor binaryLabels(int[][] labels) {
        return binaryLabels(BinaryLabelMatrix.from(labels));
    }

    static GradTensor binaryLabels(boolean[][] labels) {
        return binaryLabels(BinaryLabelMatrix.from(labels));
    }

    static GradTensor binaryLabels(float[][] labels) {
        return binaryLabels(BinaryLabelMatrix.from(labels));
    }

    static GradTensor binaryLabels(BinaryLabelMatrix matrix) {
        Objects.requireNonNull(matrix, "matrix must not be null");
        return GradTensor.of(matrix.values(), matrix.rows(), matrix.columns());
    }

    static DataLoader.TensorDataset classificationDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        requireInputBatchSize(inputs, labels.length);
        return tensorDataset(inputs, classLabels(labels));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[] labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        requireInputBatchSize(inputs, labels.length);
        return tensorDataset(inputs, binaryLabels(labels));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, int[][] labels) {
        return binaryDataset(inputs, BinaryLabelMatrix.from(labels));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, boolean[][] labels) {
        return binaryDataset(inputs, BinaryLabelMatrix.from(labels));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, float[][] labels) {
        return binaryDataset(inputs, BinaryLabelMatrix.from(labels));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, BinaryLabelMatrix matrix) {
        Objects.requireNonNull(matrix, "matrix must not be null");
        return binaryDataset(inputs, binaryLabels(matrix));
    }

    static DataLoader.TensorDataset binaryDataset(GradTensor inputs, GradTensor labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        requireInputBatchSize(inputs, labels.shape()[0]);
        return tensorDataset(inputs, labels);
    }

    private static void requireInputBatchSize(GradTensor inputs, long expectedBatchSize) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (inputs.shape().length == 0) {
            throw new IllegalArgumentException("inputs must include a batch dimension");
        }
        if (inputs.shape()[0] != expectedBatchSize) {
            throw new IllegalArgumentException(
                    "inputs and labels must have same batch dimension, got: "
                            + inputs.shape()[0] + " vs " + expectedBatchSize);
        }
    }
}
