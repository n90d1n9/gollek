package tech.kayys.gollek.train.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Objects;
import java.util.function.ToIntFunction;

final class SequenceLengthSupport {
    private SequenceLengthSupport() {
    }

    static int sequenceLength(GradTensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        long[] shape = tensor.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException("tensor must have rank >= 1 to infer sequence length");
        }
        return Math.toIntExact(shape[0]);
    }

    static <T> int[] lengths(Dataset<? extends T> dataset, ToIntFunction<? super T> lengthExtractor) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(lengthExtractor, "lengthExtractor must not be null");
        int[] lengths = new int[dataset.size()];
        for (int i = 0; i < lengths.length; i++) {
            T sample = Objects.requireNonNull(dataset.get(i), "dataset sample must not be null");
            int length = lengthExtractor.applyAsInt(sample);
            if (length < 0) {
                throw new IllegalArgumentException("sequence lengths must be non-negative, got: " + length);
            }
            lengths[i] = length;
        }
        return lengths;
    }

    static ToIntFunction<Dataset.Sample> sampleInputLength() {
        return sample -> sequenceLength(Objects.requireNonNull(sample, "sample must not be null").input());
    }

    static ToIntFunction<Dataset.Sample> sampleLabelLength() {
        return sample -> sequenceLength(Objects.requireNonNull(sample, "sample must not be null").label());
    }

    static ToIntFunction<Dataset.Pair<GradTensor, GradTensor>> tensorPairInputLength() {
        return pair -> sequenceLength(Objects.requireNonNull(pair, "pair must not be null").left());
    }

    static ToIntFunction<Dataset.Pair<GradTensor, GradTensor>> tensorPairLabelLength() {
        return pair -> sequenceLength(Objects.requireNonNull(pair, "pair must not be null").right());
    }
}
