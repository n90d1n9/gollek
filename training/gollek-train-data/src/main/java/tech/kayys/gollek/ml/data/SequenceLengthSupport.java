package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Objects;
import java.util.function.ToIntFunction;
import tech.kayys.gollek.train.data.internal.DataLoaderSequenceLengthRules;

final class SequenceLengthSupport {
    private SequenceLengthSupport() {
    }

    static int sequenceLength(GradTensor tensor) {
        return DataLoaderSequenceLengthRules.sequenceLength(tensor);
    }

    static <T> int[] lengths(Dataset<? extends T> dataset, ToIntFunction<? super T> lengthExtractor) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        return DataLoaderSequenceLengthRules.lengths(dataset.size(), dataset::get, lengthExtractor);
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
