package tech.kayys.gollek.ml.train;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable container builders for JSON-shaped report snapshots.
 */
final class TrainingReportSnapshotContainers {
    private TrainingReportSnapshotContainers() {
    }

    static Map<String, Object> map(
            Map<?, ?> map,
            TrainingReportSnapshotState state,
            NestedSnapshot nestedSnapshot) {
        state.requireContainerSize("map", map.size());
        state.enter(map);
        try {
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                String key = TrainingReportSnapshotMapKeys.normalize(entry.getKey());
                TrainingReportSnapshotMapKeys.requireAbsent(copy, key);
                copy.put(key, nestedSnapshot.snapshot(entry.getValue(), state.child()));
            }
            return Collections.unmodifiableMap(copy);
        } finally {
            state.exit(map);
        }
    }

    static List<Object> array(
            Object array,
            TrainingReportSnapshotState state,
            NestedSnapshot nestedSnapshot) {
        state.enter(array);
        try {
            int length = Array.getLength(array);
            state.requireContainerSize("array", length);
            List<Object> copy = new ArrayList<>(length);
            for (int index = 0; index < length; index++) {
                copy.add(nestedSnapshot.snapshot(Array.get(array, index), state.child()));
            }
            return Collections.unmodifiableList(copy);
        } finally {
            state.exit(array);
        }
    }

    static List<Object> iterable(
            Iterable<?> iterable,
            TrainingReportSnapshotState state,
            NestedSnapshot nestedSnapshot) {
        if (iterable instanceof Collection<?> collection) {
            state.requireContainerSize("iterable", collection.size());
        }
        state.enter(iterable);
        try {
            List<Object> copy = new ArrayList<>();
            for (Object item : iterable) {
                state.requireIterableRoom(copy.size());
                copy.add(nestedSnapshot.snapshot(item, state.child()));
            }
            return Collections.unmodifiableList(copy);
        } finally {
            state.exit(iterable);
        }
    }

    @FunctionalInterface
    interface NestedSnapshot {
        Object snapshot(Object value, TrainingReportSnapshotState state);
    }
}
