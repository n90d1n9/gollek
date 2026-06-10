package tech.kayys.gollek.ml.data;

import java.util.List;
import tech.kayys.gollek.train.data.internal.DataLoaderBatchingRules;

final class TensorBatching {
    private TensorBatching() {
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
