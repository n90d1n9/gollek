package tech.kayys.gollek.ml.train;

/**
 * Safety limits for freezing Java report metadata into immutable JSON-shaped snapshots.
 */
record TrainingReportSnapshotPolicy(int maxDepth, int maxContainerSize) {
    static final int DEFAULT_MAX_DEPTH = 256;
    static final int DEFAULT_MAX_CONTAINER_SIZE = 100_000;

    TrainingReportSnapshotPolicy {
        if (maxDepth <= 0) {
            throw new IllegalArgumentException("maxDepth must be positive");
        }
        if (maxContainerSize <= 0) {
            throw new IllegalArgumentException("maxContainerSize must be positive");
        }
    }

    static TrainingReportSnapshotPolicy defaultPolicy() {
        return new TrainingReportSnapshotPolicy(DEFAULT_MAX_DEPTH, DEFAULT_MAX_CONTAINER_SIZE);
    }
}
