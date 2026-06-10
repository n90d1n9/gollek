package tech.kayys.gollek.train.data.internal;

public final class DataLoaderCountRules {
    private DataLoaderCountRules() {
    }

    public static int requireDatasetSize(int datasetSize) {
        return requireNonNegative(datasetSize, "datasetSize");
    }

    public static int requireSampleCount(int sampleCount) {
        return requireNonNegative(sampleCount, "sampleCount");
    }

    public static int requireBatchCount(int batchCount) {
        return requireNonNegative(batchCount, "batchCount");
    }

    public static int requireNonNegative(int value, String fieldName) {
        if (value < 0) {
            throw new IllegalArgumentException(fieldName + " must be non-negative, got: " + value);
        }
        return value;
    }
}
