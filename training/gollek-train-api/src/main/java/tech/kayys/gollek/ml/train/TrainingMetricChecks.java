package tech.kayys.gollek.ml.train;

import java.util.Arrays;
import tech.kayys.gollek.ml.autograd.GradTensor;

/** Shared validation and shape helpers for built-in training metrics. */
final class TrainingMetricChecks {
    private TrainingMetricChecks() {
    }

    static float requireFiniteLogitThreshold(float logitThreshold) {
        if (!Float.isFinite(logitThreshold)) {
            throw new IllegalArgumentException("logitThreshold must be finite, got: " + logitThreshold);
        }
        return logitThreshold;
    }

    static double requireUnitInterval(String name, double value) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(name + " must be finite and in [0, 1], got: " + value);
        }
        return value;
    }

    static void requireSameShape(String metricName, GradTensor predictions, GradTensor targets) {
        long[] predictionShape = predictions.shape();
        long[] targetShape = targets.shape();
        if (!Arrays.equals(predictionShape, targetShape)) {
            throw new IllegalArgumentException(
                    metricName + " expects predictions and targets with the same shape, got "
                            + Arrays.toString(predictionShape) + " vs " + Arrays.toString(targetShape));
        }
    }

    static void requireSameElementCount(String metricName, GradTensor predictions, GradTensor targets) {
        if (predictions.numel() != targets.numel()) {
            throw new IllegalArgumentException(
                    metricName + " expects predictions and targets with the same element count, got "
                            + predictions.numel() + " vs " + targets.numel());
        }
    }

    static int multiLabelSampleCount(long[] shape) {
        if (shape.length == 0) {
            return 1;
        }
        return Math.toIntExact(shape[0]);
    }

    static int multiLabelLabelsPerSample(long[] shape) {
        if (shape.length <= 1) {
            return 1;
        }
        long labels = 1;
        for (int i = 1; i < shape.length; i++) {
            labels = Math.multiplyExact(labels, shape[i]);
        }
        return Math.toIntExact(labels);
    }

    static boolean binaryTarget(float value) {
        if (Math.abs(value) <= 1e-6f) {
            return false;
        }
        if (Math.abs(value - 1.0f) <= 1e-6f) {
            return true;
        }
        throw new IllegalArgumentException("binary targets must contain only 0.0 or 1.0, got: " + value);
    }

    static int argmax(float[] values, int offset, int length) {
        int best = 0;
        float bestValue = values[offset];
        for (int i = 1; i < length; i++) {
            float value = values[offset + i];
            if (value > bestValue) {
                best = i;
                bestValue = value;
            }
        }
        return best;
    }

    static boolean containsTopK(float[] values, int offset, int length, int targetIndex, int k) {
        float targetValue = values[offset + targetIndex];
        int strictlyGreater = 0;
        int equalBeforeTarget = 0;
        for (int i = 0; i < length; i++) {
            if (i == targetIndex) {
                continue;
            }
            float value = values[offset + i];
            if (value > targetValue) {
                strictlyGreater++;
            } else if (value == targetValue && i < targetIndex) {
                equalBeforeTarget++;
            }
        }
        return strictlyGreater + equalBeforeTarget < k;
    }

    static int targetClass(GradTensor targets, int row, int expectedBatch, int classes) {
        long[] targetShape = targets.shape();
        float[] targetData = targets.data();
        if (targetShape.length == 1 && targetShape[0] == expectedBatch) {
            return checkedTargetClass((int) targetData[row], classes);
        }
        if (targetShape.length == 2 && targetShape[0] == expectedBatch && targetShape[1] == classes) {
            return argmax(targetData, row * classes, classes);
        }
        throw new IllegalArgumentException(
                "classification metric expects targets shaped [batch] class indices or [batch, classes] one-hot labels, got "
                        + Arrays.toString(targetShape));
    }

    static int checkedTargetClass(int targetClass, int classes) {
        if (targetClass < 0 || targetClass >= classes) {
            throw new IllegalArgumentException(
                    "target class " + targetClass + " out of range [0, " + (classes - 1) + "]");
        }
        return targetClass;
    }
}
