package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Publishes rolling loss-window statistics for stability monitoring.
 */
final class TrainerLossWindowStatsMetadata {
    private static final int DEFAULT_WINDOW_SIZE = 5;
    private static final int MIN_POINTS = 2;

    private TrainerLossWindowStatsMetadata() {
    }

    static void putLatest(Map<String, Object> metadata, List<Map<String, Object>> rows) {
        putLatest(metadata, rows, "trainLoss", "latestTrainLoss");
        putLatest(metadata, rows, "validationLoss", "latestValidationLoss");
    }

    static void putEpoch(Map<String, Object> row, List<Map<String, Object>> rows, String lossKey) {
        put(row, latestValuesThroughEpoch(row, rows, lossKey), lossKey);
    }

    private static void putLatest(
            Map<String, Object> metadata,
            List<Map<String, Object>> rows,
            String lossKey,
            String prefix) {
        put(metadata, latestValues(rows, lossKey), prefix);
    }

    private static void put(Map<String, Object> metadata, List<Double> values, String prefix) {
        String availabilityKey = prefix + "WindowStatsAvailable";
        String windowKey = prefix + "WindowStatsSize";
        String meanKey = prefix + "WindowMean";
        String stdDevKey = prefix + "WindowStdDev";
        String variationKey = prefix + "WindowCoefficientOfVariation";
        boolean available = values.size() >= MIN_POINTS;
        metadata.put(availabilityKey, available);
        if (!available) {
            metadata.remove(windowKey);
            metadata.remove(meanKey);
            metadata.remove(stdDevKey);
            metadata.remove(variationKey);
            return;
        }
        double mean = mean(values);
        double stdDev = stdDev(values, mean);
        metadata.put(windowKey, values.size());
        metadata.put(meanKey, mean);
        metadata.put(stdDevKey, stdDev);
        metadata.put(variationKey, TrainerOptimizationMetadata.ratio(stdDev, Math.abs(mean)));
    }

    private static List<Double> latestValues(List<Map<String, Object>> rows, String lossKey) {
        List<Double> values = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            Double loss = finiteNumber(row.get(lossKey));
            if (loss != null) {
                values.add(loss);
            }
        }
        int from = Math.max(0, values.size() - DEFAULT_WINDOW_SIZE);
        return values.subList(from, values.size());
    }

    private static List<Double> latestValuesThroughEpoch(
            Map<String, Object> currentRow,
            List<Map<String, Object>> rows,
            String lossKey) {
        Double currentLoss = finiteNumber(currentRow.get(lossKey));
        if (currentLoss == null) {
            return List.of();
        }
        double currentEpoch = epoch(currentRow.get("epoch"), Double.MAX_VALUE);
        List<Double> values = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            if (!row.containsKey(lossKey)) {
                continue;
            }
            double candidateEpoch = epoch(row.get("epoch"), Double.MAX_VALUE);
            if (candidateEpoch > currentEpoch) {
                continue;
            }
            Double loss = finiteNumber(row.get(lossKey));
            if (loss != null) {
                values.add(loss);
            }
        }
        if (!rows.contains(currentRow)) {
            values.add(currentLoss);
        }
        int from = Math.max(0, values.size() - DEFAULT_WINDOW_SIZE);
        return values.subList(from, values.size());
    }

    private static double epoch(Object value, double fallback) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : fallback;
        }
        return fallback;
    }

    private static double mean(List<Double> values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.size();
    }

    private static double stdDev(List<Double> values, double mean) {
        double sumSquaredDeviation = 0.0;
        for (double value : values) {
            double deviation = value - mean;
            sumSquaredDeviation += deviation * deviation;
        }
        return Math.sqrt(sumSquaredDeviation / values.size());
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }
}
