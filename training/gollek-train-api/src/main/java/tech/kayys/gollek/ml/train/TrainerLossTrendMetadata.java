package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Publishes loss deltas between epochs for plateau/regression monitoring.
 */
final class TrainerLossTrendMetadata {
    private TrainerLossTrendMetadata() {
    }

    static void putLatest(Map<String, Object> metadata, List<Map<String, Object>> rows) {
        putLatest(metadata, rows, "trainLoss", "latestTrainLossDelta");
        putLatest(metadata, rows, "validationLoss", "latestValidationLossDelta");
    }

    static void putEpoch(
            Map<String, Object> row,
            Map<String, Object> previousRow,
            String lossKey,
            String deltaKey,
            String improvedKey) {
        Double current = finiteNumber(row.get(lossKey));
        Double previous = previousRow == null ? null : finiteNumber(previousRow.get(lossKey));
        if (current == null || previous == null) {
            row.remove(deltaKey);
            row.remove(improvedKey);
            return;
        }
        double delta = current - previous;
        row.put(deltaKey, delta);
        row.put(improvedKey, delta < 0.0);
    }

    private static void putLatest(
            Map<String, Object> metadata,
            List<Map<String, Object>> rows,
            String lossKey,
            String deltaKey) {
        String availabilityKey = deltaKey + "Available";
        String improvedKey = deltaKey.replace("Delta", "Improved");
        Double previous = null;
        Double latest = null;
        for (Map<String, Object> row : rows) {
            Double value = finiteNumber(row.get(lossKey));
            if (value == null) {
                continue;
            }
            previous = latest;
            latest = value;
        }
        boolean available = previous != null && latest != null;
        metadata.put(availabilityKey, available);
        if (!available) {
            metadata.remove(deltaKey);
            metadata.remove(improvedKey);
            return;
        }
        double delta = latest - previous;
        metadata.put(deltaKey, delta);
        metadata.put(improvedKey, delta < 0.0);
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }
}
