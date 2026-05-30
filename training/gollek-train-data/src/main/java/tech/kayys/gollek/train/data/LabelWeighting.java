package tech.kayys.gollek.train.data;

import java.util.Arrays;
import java.util.Objects;

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
        return DistributionDiagnostics.balancedClassWeights(counts, labels.length);
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
        return DistributionDiagnostics.positiveWeight(positives, labels.length);
    }

    static float binaryPositiveWeight(boolean... labels) {
        Objects.requireNonNull(labels, "labels must not be null");
        int positives = 0;
        for (boolean label : labels) {
            positives += label ? 1 : 0;
        }
        return DistributionDiagnostics.positiveWeight(positives, labels.length);
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
        return DistributionDiagnostics.positiveWeight(positives, labels.length);
    }

    static float[] positiveWeights(BinaryLabelMatrix matrix) {
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
            weights[column] = DistributionDiagnostics.positiveWeight(positives[column], matrix.rows());
        }
        return weights;
    }

    static float[] positiveWeights(int[][] labels) {
        return positiveWeights(BinaryLabelMatrix.from(labels));
    }

    static float[] positiveWeights(boolean[][] labels) {
        return positiveWeights(BinaryLabelMatrix.from(labels));
    }

    static float[] positiveWeights(float[][] labels) {
        return positiveWeights(BinaryLabelMatrix.from(labels));
    }

    static float[] multiLabelBalancedSampleWeights(BinaryLabelMatrix matrix) {
        int[] positives = new int[matrix.columns()];
        int[] rowPositiveCounts = new int[matrix.rows()];
        for (int row = 0; row < matrix.rows(); row++) {
            int offset = row * matrix.columns();
            for (int column = 0; column < matrix.columns(); column++) {
                if (matrix.values()[offset + column] >= 0.5f) {
                    positives[column]++;
                    rowPositiveCounts[row]++;
                }
            }
        }

        int activeColumns = 0;
        for (int count : positives) {
            if (count > 0) {
                activeColumns++;
            }
        }
        if (activeColumns == 0) {
            float[] weights = new float[matrix.rows()];
            Arrays.fill(weights, 1.0f);
            return weights;
        }

        float[] columnWeights = new float[matrix.columns()];
        float minimumActiveWeight = Float.POSITIVE_INFINITY;
        for (int column = 0; column < matrix.columns(); column++) {
            if (positives[column] == 0) {
                continue;
            }
            columnWeights[column] = matrix.rows() / (float) (activeColumns * positives[column]);
            minimumActiveWeight = Math.min(minimumActiveWeight, columnWeights[column]);
        }

        float[] sampleWeights = new float[matrix.rows()];
        for (int row = 0; row < matrix.rows(); row++) {
            if (rowPositiveCounts[row] == 0) {
                sampleWeights[row] = minimumActiveWeight;
                continue;
            }
            float sum = 0.0f;
            int offset = row * matrix.columns();
            for (int column = 0; column < matrix.columns(); column++) {
                if (matrix.values()[offset + column] >= 0.5f) {
                    sum += columnWeights[column];
                }
            }
            sampleWeights[row] = sum / rowPositiveCounts[row];
        }
        return sampleWeights;
    }

    static float[] multiLabelBalancedSampleWeights(int[][] labels) {
        return multiLabelBalancedSampleWeights(BinaryLabelMatrix.from(labels));
    }

    static float[] multiLabelBalancedSampleWeights(boolean[][] labels) {
        return multiLabelBalancedSampleWeights(BinaryLabelMatrix.from(labels));
    }

    static float[] multiLabelBalancedSampleWeights(float[][] labels) {
        return multiLabelBalancedSampleWeights(BinaryLabelMatrix.from(labels));
    }
}
