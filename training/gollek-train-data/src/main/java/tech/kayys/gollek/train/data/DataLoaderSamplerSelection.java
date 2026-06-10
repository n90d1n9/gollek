package tech.kayys.gollek.train.data;

import tech.kayys.gollek.train.data.internal.DataLoaderSamplerSelectionRules;

final class DataLoaderSamplerSelection {
    private DataLoaderSamplerSelection() {
    }

    static IndexSampler requireSampler(IndexSampler sampler) {
        return DataLoaderSamplerSelectionRules.requireSampler(sampler);
    }

    static BatchSampler requireBatchSampler(BatchSampler batchSampler) {
        return DataLoaderSamplerSelectionRules.requireBatchSampler(batchSampler);
    }

    static void requireExclusive(IndexSampler sampler, BatchSampler batchSampler) {
        DataLoaderSamplerSelectionRules.requireExclusive(sampler, batchSampler);
    }
}
