package tech.kayys.gollek.train.data;

import java.util.List;
import java.util.Objects;
import tech.kayys.gollek.train.data.internal.DataLoaderBatchingRules;

final class TensorBatching {
    private TensorBatching() {
    }

    static List<Integer> epochIndices(
            DataLoader.TensorDatasetAdapter dataset,
            IndexSampler sampler,
            boolean shuffle,
            Long shuffleSeed) {
        return epochIndices(dataset, sampler, shuffle, shuffleSeed, false, 0L);
    }

    static List<Integer> epochIndices(
            DataLoader.TensorDatasetAdapter dataset,
            IndexSampler sampler,
            boolean shuffle,
            Long shuffleSeed,
            boolean reshuffleEachEpoch,
            long epoch) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        int datasetSize = dataset.size();
        return epochIndices(datasetSize, sampler, shuffle, shuffleSeed, reshuffleEachEpoch, epoch);
    }

    static List<Integer> epochIndices(
            int datasetSize,
            IndexSampler sampler,
            boolean shuffle,
            Long shuffleSeed,
            boolean reshuffleEachEpoch,
            long epoch) {
        return DataLoaderBatchingRules.epochIndices(
                datasetSize,
                sampler == null ? null : sampler::sample,
                shuffle,
                shuffleSeed,
                reshuffleEachEpoch,
                epoch);
    }

    static int batchCount(int sampleCount, int batchSize, boolean dropLast) {
        return DataLoaderBatchingRules.batchCount(sampleCount, batchSize, dropLast);
    }

    static List<List<Integer>> fixedBatches(List<Integer> indices, int batchSize, boolean dropLast) {
        return DataLoaderBatchingRules.fixedBatches(indices, batchSize, dropLast);
    }
}
