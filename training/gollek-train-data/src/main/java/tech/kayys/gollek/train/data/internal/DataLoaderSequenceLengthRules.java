package tech.kayys.gollek.train.data.internal;

import java.util.Objects;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;
import tech.kayys.gollek.ml.autograd.GradTensor;

public final class DataLoaderSequenceLengthRules {
    private DataLoaderSequenceLengthRules() {
    }

    public static int sequenceLength(GradTensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        long[] shape = tensor.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException("tensor must have rank >= 1 to infer sequence length");
        }
        return Math.toIntExact(shape[0]);
    }

    public static <T> int[] lengths(
            int datasetSize,
            IntFunction<? extends T> rowReader,
            ToIntFunction<? super T> lengthExtractor) {
        DataLoaderCountRules.requireDatasetSize(datasetSize);
        Objects.requireNonNull(rowReader, "rowReader must not be null");
        Objects.requireNonNull(lengthExtractor, "lengthExtractor must not be null");
        int[] lengths = new int[datasetSize];
        for (int i = 0; i < lengths.length; i++) {
            T sample = Objects.requireNonNull(rowReader.apply(i), "dataset sample must not be null");
            int length = lengthExtractor.applyAsInt(sample);
            if (length < 0) {
                throw new IllegalArgumentException("sequence lengths must be non-negative, got: " + length);
            }
            lengths[i] = length;
        }
        return lengths;
    }
}
