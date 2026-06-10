package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Map-key normalization for JSON-shaped report snapshots.
 */
final class TrainingReportSnapshotMapKeys {
    private TrainingReportSnapshotMapKeys() {
    }

    static String normalize(Object key) {
        Object snapshot = TrainingReportSnapshotScalars.snapshot(key);
        return snapshot == null ? "null" : String.valueOf(snapshot);
    }

    static void requireAbsent(Map<String, ?> values, String key) {
        if (values.containsKey(key)) {
            throw new IllegalArgumentException(
                    "report snapshot contains duplicate map key after string normalization: " + key);
        }
    }
}
