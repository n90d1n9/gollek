package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable snapshot helper for report payloads and report-derived evidence.
 */
final class TrainingReportSnapshots {
    private TrainingReportSnapshots() {
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> immutableMap(Map<?, ?> map) {
        return immutableMap(map, TrainingReportSnapshotPolicy.defaultPolicy());
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> immutableMap(Map<?, ?> map, TrainingReportSnapshotPolicy policy) {
        Objects.requireNonNull(map, "map must not be null");
        Object snapshot = immutableSnapshot(map, policy);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    static Object immutableSnapshot(Object value) {
        return immutableSnapshot(value, TrainingReportSnapshotPolicy.defaultPolicy());
    }

    static Object immutableSnapshot(Object value, TrainingReportSnapshotPolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        return immutableSnapshot(value, TrainingReportSnapshotState.root(policy));
    }

    private static Object immutableSnapshot(Object value, TrainingReportSnapshotState state) {
        state.requireDepth();
        if (TrainingReportSnapshotScalars.isScalar(value)) {
            return TrainingReportSnapshotScalars.snapshot(value);
        }
        if (TrainingReportSnapshotOptionals.isOptional(value)) {
            return TrainingReportSnapshotOptionals.snapshot(value, nested -> immutableSnapshot(nested, state.child()));
        }
        if (value instanceof Map<?, ?> map) {
            return TrainingReportSnapshotContainers.map(map, state, TrainingReportSnapshots::immutableSnapshot);
        }
        if (value != null && value.getClass().isArray()) {
            return TrainingReportSnapshotContainers.array(value, state, TrainingReportSnapshots::immutableSnapshot);
        }
        if (value instanceof Iterable<?> iterable) {
            return TrainingReportSnapshotContainers.iterable(iterable, state, TrainingReportSnapshots::immutableSnapshot);
        }
        return TrainingReportSnapshotScalars.snapshot(value);
    }
}
