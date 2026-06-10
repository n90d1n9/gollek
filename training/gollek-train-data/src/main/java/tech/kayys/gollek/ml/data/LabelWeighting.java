package tech.kayys.gollek.ml.data;

import java.util.Objects;

/**
 * Label weighting utilities for the legacy {@code tech.kayys.gollek.ml.data} facade.
 */
final class LabelWeighting {
    private LabelWeighting() {
    }

    static float[] classWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int maxLabel = -1;
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            maxLabel = Math.max(maxLabel, label);
        }
        return classWeightsFor(maxLabel + 1, labels);
    }

    static float[] classWeightsFor(int numClasses, int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (numClasses <= 0) {
            throw new IllegalArgumentException("numClasses must be positive, got: " + numClasses);
        }
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int[] counts = new int[numClasses];
        for (int label : labels) {
            if (label < 0 || label >= numClasses) {
                throw new IllegalArgumentException(
                        "class label " + label + " out of range [0, " + (numClasses - 1) + "]");
            }
            counts[label]++;
        }
        return balancedClassWeights(counts, labels.length);
    }

    static float[] classBalancedSampleWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int maxLabel = -1;
        for (int label : labels) {
            if (label < 0) {
                throw new IllegalArgumentException("class labels must be non-negative, got: " + label);
            }
            maxLabel = Math.max(maxLabel, label);
        }
        float[] classWeights = classWeightsFor(maxLabel + 1, labels);
        float[] sampleWeights = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            sampleWeights[i] = classWeights[labels[i]];
        }
        return sampleWeights;
    }

    static float[] binaryBalancedSampleWeights(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        for (int label : labels) {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + label);
            }
        }
        return classBalancedSampleWeights(labels);
    }

    static float binaryPositiveWeight(int... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (int label : labels) {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("binary labels must be 0 or 1, got: " + label);
            }
            positives += label;
        }
        return positiveWeight(positives, labels.length);
    }

    static float binaryPositiveWeight(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (boolean label : labels) {
            positives += label ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    static float binaryPositiveWeight(float... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (float label : labels) {
            if (Math.abs(label) > 1e-6f && Math.abs(label - 1.0f) > 1e-6f) {
                throw new IllegalArgumentException("binary labels must be 0.0 or 1.0, got: " + label);
            }
            positives += label >= 0.5f ? 1 : 0;
        }
        return positiveWeight(positives, labels.length);
    }

    static float[] positiveWeights(BinaryLabelMatrix matrix) {
        Objects.requireNonNull(matrix, "matrix must not be null");
        int[] positives = new int[matrix.columns()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int column = 0; column < matrix.columns(); column++) {
                if (matrix.values()[row * matrix.columns() + column] >= 0.5f) {
                    positives[column]++;
                }
            }
        }
        float[] weights = new float[matrix.columns()];
        for (int column = 0; column < matrix.columns(); column++) {
            weights[column] = positiveWeight(positives[column], matrix.rows());
        }
        return weights;
    }

    static float positiveWeight(int positives, int total) {
        if (total <= 0) {
            throw new IllegalArgumentException("labels must contain at least one value");
        }
        int negatives = total - positives;
        if (positives == 0 || negatives == 0) {
            return 1.0f;
        }
        return negatives / (float) positives;
    }

    private static float[] balancedClassWeights(int[] counts, int total) {
        float[] weights = new float[counts.length];
        for (int i = 0; i < counts.length; i++) {
            weights[i] = counts[i] == 0 ? 1.0f : total / (float) (counts.length * counts[i]);
        }
        return weights;
    }
}
