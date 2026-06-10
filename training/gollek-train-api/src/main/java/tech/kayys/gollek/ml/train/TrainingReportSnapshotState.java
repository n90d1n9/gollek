package tech.kayys.gollek.ml.train;

import java.util.IdentityHashMap;

/**
 * Recursion state and safety checks for immutable report snapshots.
 */
final class TrainingReportSnapshotState {
    private final TrainingReportSnapshotPolicy policy;
    private final IdentityHashMap<Object, Boolean> seen;
    private final int depth;

    private TrainingReportSnapshotState(
            TrainingReportSnapshotPolicy policy,
            IdentityHashMap<Object, Boolean> seen,
            int depth) {
        this.policy = policy;
        this.seen = seen;
        this.depth = depth;
    }

    static TrainingReportSnapshotState root(TrainingReportSnapshotPolicy policy) {
        return new TrainingReportSnapshotState(policy, new IdentityHashMap<>(), 0);
    }

    TrainingReportSnapshotState child() {
        return new TrainingReportSnapshotState(policy, seen, depth + 1);
    }

    void requireDepth() {
        if (depth > policy.maxDepth()) {
            throw new IllegalArgumentException("report snapshot exceeds maximum nesting depth: " + policy.maxDepth());
        }
    }

    void requireContainerSize(String type, int size) {
        if (size > policy.maxContainerSize()) {
            throw new IllegalArgumentException(
                    "report snapshot " + type + " exceeds maximum size: " + policy.maxContainerSize());
        }
    }

    void requireIterableRoom(int size) {
        if (size >= policy.maxContainerSize()) {
            throw new IllegalArgumentException(
                    "report snapshot iterable exceeds maximum size: " + policy.maxContainerSize());
        }
    }

    void enter(Object value) {
        if (seen.put(value, Boolean.TRUE) != null) {
            throw new IllegalArgumentException("report snapshot contains a cyclic object graph");
        }
    }

    void exit(Object value) {
        seen.remove(value);
    }
}
