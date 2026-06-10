package tech.kayys.gollek.train.data.internal;

public final class DataLoaderBatchSizeRules {
    private DataLoaderBatchSizeRules() {
    }

    public static int requirePositive(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        return batchSize;
    }
}
