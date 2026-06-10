package tech.kayys.gollek.train.data;

import tech.kayys.gollek.train.data.internal.DataLoaderBatchSizeRules;

final class DataLoaderBatchSizes {
    private DataLoaderBatchSizes() {
    }

    static int requirePositive(int batchSize) {
        return DataLoaderBatchSizeRules.requirePositive(batchSize);
    }
}
