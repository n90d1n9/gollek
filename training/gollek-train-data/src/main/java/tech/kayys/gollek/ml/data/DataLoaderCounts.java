package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.train.data.internal.DataLoaderCountRules;

final class DataLoaderCounts {
    private DataLoaderCounts() {
    }

    static int requireDatasetSize(int datasetSize) {
        return DataLoaderCountRules.requireDatasetSize(datasetSize);
    }

    static int requireSampleCount(int sampleCount) {
        return DataLoaderCountRules.requireSampleCount(sampleCount);
    }

    static int requireBatchCount(int batchCount) {
        return DataLoaderCountRules.requireBatchCount(batchCount);
    }

    static int requireNonNegative(int value, String fieldName) {
        return DataLoaderCountRules.requireNonNegative(value, fieldName);
    }
}
