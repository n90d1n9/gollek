package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Objects;
import java.util.function.ToIntFunction;

final class TensorDataLoaderBuilderState {
    private final DataLoader.TensorDatasetAdapter dataset;
    private int batchSize = 32;
    private boolean shuffle;
    private boolean dropLast;
    private Long shuffleSeed;
    private boolean reshuffleEachEpoch;
    private long initialEpoch;
    private IndexSampler sampler;
    private BatchSampler batchSampler;
    private DataLoader.CollateFn collateFn;

    TensorDataLoaderBuilderState(DataLoader.TensorDatasetAdapter dataset) {
        this.dataset = Objects.requireNonNull(dataset, "dataset must not be null");
    }

    int batchSize() {
        return batchSize;
    }

    boolean dropLast() {
        return dropLast;
    }

    void batchSize(int batchSize) {
        this.batchSize = DataLoaderBatchSizes.requirePositive(batchSize);
    }

    void shuffle(boolean shuffle) {
        this.shuffle = shuffle;
        if (!shuffle) {
            this.reshuffleEachEpoch = false;
        }
    }

    void dropLast(boolean dropLast) {
        this.dropLast = dropLast;
    }

    void seed(long seed) {
        this.shuffleSeed = seed;
    }

    void reshuffleEachEpoch(boolean reshuffleEachEpoch) {
        this.reshuffleEachEpoch = reshuffleEachEpoch;
        if (reshuffleEachEpoch) {
            this.shuffle = true;
        }
    }

    void initialEpoch(long initialEpoch) {
        DataLoaderEpochs.requireInitialEpoch(initialEpoch);
        this.initialEpoch = initialEpoch;
    }

    void sampler(IndexSampler sampler) {
        this.sampler = DataLoaderSamplerSelection.requireSampler(sampler);
        this.batchSampler = null;
    }

    void batchSampler(BatchSampler batchSampler) {
        this.batchSampler = DataLoaderSamplerSelection.requireBatchSampler(batchSampler);
        this.sampler = null;
    }

    void collateFn(DataLoader.CollateFn collateFn) {
        this.collateFn = collateFn;
    }

    int[] tensorDatasetLengths(ToIntFunction<GradTensor[]> lengthExtractor) {
        Objects.requireNonNull(lengthExtractor, "lengthExtractor must not be null");
        int[] lengths = new int[dataset.size()];
        for (int i = 0; i < lengths.length; i++) {
            GradTensor[] sample = Objects.requireNonNull(dataset.get(i), "dataset sample must not be null");
            int length = lengthExtractor.applyAsInt(sample);
            if (length < 0) {
                throw new IllegalArgumentException("sequence lengths must be non-negative, got: " + length);
            }
            lengths[i] = length;
        }
        return lengths;
    }

    TensorDataLoaderConfig config() {
        return new TensorDataLoaderConfig(
                dataset,
                batchSize,
                shuffle,
                dropLast,
                shuffleSeed,
                reshuffleEachEpoch,
                initialEpoch,
                sampler,
                batchSampler,
                collateFn);
    }
}
