package tech.kayys.gollek.train.data.internal;

import java.util.Objects;
import tech.kayys.gollek.ml.autograd.GradTensor;

public final class DataLoaderTensorBatchValidationRules {
    private DataLoaderTensorBatchValidationRules() {
    }

    public static GradTensor requireTensor(String name, GradTensor tensor) {
        return Objects.requireNonNull(tensor, name + " must not be null");
    }

    public static int[] copyLengths(String name, int[] lengths) {
        return Objects.requireNonNull(lengths, name + " must not be null").clone();
    }

    public static void requireCompatibleBatchDimensions(GradTensor inputs, GradTensor labels) {
        long inputBatchSize = batchSize("inputs", inputs);
        long labelBatchSize = batchSize("labels", labels);
        if (inputBatchSize != labelBatchSize) {
            throw new IllegalArgumentException(
                    "inputs and labels must have the same batch dimension, got: "
                            + inputBatchSize + " vs " + labelBatchSize);
        }
    }

    public static void validatePadded(String name, GradTensor values, GradTensor mask, int[] lengths) {
        long[] valueShape = values.shape();
        if (valueShape.length < 2) {
            throw new IllegalArgumentException(name + " must include batch and padded sequence dimensions");
        }
        if (valueShape[0] != lengths.length) {
            throw new IllegalArgumentException(name + " batch dimension must match lengths");
        }
        long[] maskShape = mask.shape();
        if (maskShape.length != 2 || maskShape[0] != valueShape[0]) {
            throw new IllegalArgumentException(name + " mask must be shaped [batch, maxLength]");
        }
        long maxLength = valueShape[1];
        if (maskShape[1] != maxLength) {
            throw new IllegalArgumentException(name + " mask length must match padded sequence length");
        }
        for (int length : lengths) {
            if (length < 0 || length > maxLength) {
                throw new IllegalArgumentException(name + " lengths must be within the padded sequence length");
            }
        }
    }

    public static int paddedLength(String name, GradTensor values) {
        try {
            return Math.toIntExact(values.shape()[1]);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(name + " padded sequence length is too large", e);
        }
    }

    private static long batchSize(String name, GradTensor tensor) {
        long[] shape = tensor.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException(name + " must include a batch dimension");
        }
        return shape[0];
    }
}
