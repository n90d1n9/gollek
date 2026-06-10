package tech.kayys.gollek.ml.data;

import java.util.Objects;

record TensorDataLoaderConfig(
        DataLoader.TensorDatasetAdapter dataset,
        int batchSize,
        boolean shuffle,
        boolean dropLast,
        Long shuffleSeed,
        boolean reshuffleEachEpoch,
        long initialEpoch,
        IndexSampler sampler,
        BatchSampler batchSampler,
        DataLoader.CollateFn collateFn) {

    TensorDataLoaderConfig {
        dataset = Objects.requireNonNull(dataset, "dataset must not be null");
        batchSize = DataLoaderBatchSizes.requirePositive(batchSize);
        DataLoaderEpochs.requireInitialEpoch(initialEpoch);
        DataLoaderSamplerSelection.requireExclusive(sampler, batchSampler);
        collateFn = collateFn != null ? collateFn : TensorCollators.defaultPairCollate();
    }
}
