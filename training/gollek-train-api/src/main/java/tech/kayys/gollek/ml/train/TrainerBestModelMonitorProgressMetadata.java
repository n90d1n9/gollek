package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Publishes row-level progress for the configured best-model monitor.
 */
final class TrainerBestModelMonitorProgressMetadata {
    private TrainerBestModelMonitorProgressMetadata() {
    }

    static void putEpoch(Map<String, Object> row, List<Map<String, Object>> rows) {
        Progress progress = progressThroughEpoch(row, rows);
        row.put("bestModelMonitorProgressAvailable", progress.available());
        if (!progress.available()) {
            row.remove("bestModelMonitorBestValue");
            row.remove("bestModelMonitorBestEpoch");
            row.remove("bestModelMonitorDistanceFromBest");
            row.remove("epochsSinceBestModelMonitor");
            row.remove("bestModelMonitorIsBest");
            return;
        }
        row.put("bestModelMonitorBestValue", progress.bestValue());
        row.put("bestModelMonitorBestEpoch", progress.bestEpoch());
        row.put("bestModelMonitorDistanceFromBest", distanceFromBest(
                progress.currentValue(),
                progress.bestValue(),
                progress.mode()));
        row.put("epochsSinceBestModelMonitor", Math.max(0, progress.currentEpoch() - progress.bestEpoch()));
        row.put("bestModelMonitorIsBest", progress.currentEpoch() == progress.bestEpoch());
    }

    private static Progress progressThroughEpoch(Map<String, Object> currentRow, List<Map<String, Object>> rows) {
        String monitor = monitorName(currentRow.get("bestModelMonitor"));
        CanonicalTrainer.BestModelMonitorMode mode = monitorMode(currentRow.get("bestModelMonitorMode"));
        Double currentValue = finiteNumber(currentRow.get("bestModelMonitorValue"));
        if (monitor == null || mode == null || currentValue == null) {
            return Progress.unavailable();
        }
        int currentEpoch = epoch(currentRow.get("epoch"), Integer.MAX_VALUE);
        double bestValue = Double.NaN;
        int bestEpoch = -1;
        boolean available = false;
        for (Map<String, Object> row : rows) {
            int candidateEpoch = epoch(row.get("epoch"), -1);
            if (candidateEpoch < 0 || candidateEpoch > currentEpoch || !sameMonitor(row, monitor, mode)) {
                continue;
            }
            Double candidateValue = finiteNumber(row.get("bestModelMonitorValue"));
            if (candidateValue == null) {
                continue;
            }
            if (!available || mode.isImproved(candidateValue, bestValue, 0.0)) {
                bestValue = candidateValue;
                bestEpoch = candidateEpoch;
                available = true;
            }
        }
        if (!rows.contains(currentRow)) {
            if (!available || mode.isImproved(currentValue, bestValue, 0.0)) {
                bestValue = currentValue;
                bestEpoch = currentEpoch;
                available = true;
            }
        }
        return available
                ? new Progress(true, currentEpoch, currentValue, bestEpoch, bestValue, mode)
                : Progress.unavailable();
    }

    private static boolean sameMonitor(
            Map<String, Object> row,
            String monitor,
            CanonicalTrainer.BestModelMonitorMode mode) {
        return Objects.equals(monitorName(row.get("bestModelMonitor")), monitor)
                && monitorMode(row.get("bestModelMonitorMode")) == mode;
    }

    private static double distanceFromBest(
            double currentValue,
            double bestValue,
            CanonicalTrainer.BestModelMonitorMode mode) {
        return mode == CanonicalTrainer.BestModelMonitorMode.MIN
                ? currentValue - bestValue
                : bestValue - currentValue;
    }

    private static String monitorName(Object value) {
        if (!(value instanceof String text)) {
            return null;
        }
        String trimmed = text.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    private static CanonicalTrainer.BestModelMonitorMode monitorMode(Object value) {
        if (!(value instanceof String text)) {
            return null;
        }
        try {
            return CanonicalTrainer.BestModelMonitorMode.valueOf(text.trim());
        } catch (IllegalArgumentException ignored) {
            return null;
        }
    }

    private static int epoch(Object value, int fallback) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            if (Double.isFinite(doubleValue)) {
                return number.intValue();
            }
        }
        return fallback;
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }

    private record Progress(
            boolean available,
            int currentEpoch,
            double currentValue,
            int bestEpoch,
            double bestValue,
            CanonicalTrainer.BestModelMonitorMode mode) {
        static Progress unavailable() {
            return new Progress(false, -1, Double.NaN, -1, Double.NaN, null);
        }
    }
}
