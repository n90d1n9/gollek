package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Publishes train-vs-validation loss diagnostics for overfitting monitoring.
 */
final class TrainerGeneralizationMetadata {
    private static final double TREND_EPSILON = 1e-12;

    private TrainerGeneralizationMetadata() {
    }

    static void putLatest(Map<String, Object> metadata, Double trainLoss, Double validationLoss) {
        boolean available = isFinite(trainLoss) && isFinite(validationLoss);
        metadata.put("generalizationGapAvailable", available);
        if (!available) {
            return;
        }
        metadata.put("latestGeneralizationGap", validationLoss - trainLoss);
        metadata.put("latestValidationToTrainLossRatio", TrainerOptimizationMetadata.ratio(validationLoss, trainLoss));
        metadata.put("latestValidationLossAboveTrainLoss", validationLoss > trainLoss);
    }

    static void putLatestTrend(Map<String, Object> metadata, List<Map<String, Object>> rows) {
        Double previous = null;
        Double latest = null;
        for (Map<String, Object> row : rows) {
            Double gap = generalizationGap(row);
            if (gap == null) {
                continue;
            }
            previous = latest;
            latest = gap;
        }
        boolean available = previous != null && latest != null;
        metadata.put("latestGeneralizationGapDeltaAvailable", available);
        if (!available) {
            metadata.remove("latestGeneralizationGapDelta");
            metadata.remove("latestGeneralizationGapTrend");
            metadata.remove("latestGeneralizationGapIncreasing");
            return;
        }
        putGapTrend(metadata, "latestGeneralizationGap", latest - previous);
    }

    static void putEpoch(Map<String, Object> row) {
        putEpoch(row, null);
    }

    static void putEpoch(Map<String, Object> row, Map<String, Object> previousRow) {
        Double trainLoss = finiteNumber(row.get("trainLoss"));
        Double validationLoss = finiteNumber(row.get("validationLoss"));
        if (trainLoss == null || validationLoss == null) {
            row.remove("generalizationGap");
            row.remove("validationToTrainLossRatio");
            row.remove("validationLossAboveTrainLoss");
            row.remove("generalizationGapDelta");
            row.remove("generalizationGapTrend");
            row.remove("generalizationGapIncreasing");
            return;
        }
        row.put("generalizationGap", validationLoss - trainLoss);
        row.put("validationToTrainLossRatio", TrainerOptimizationMetadata.ratio(validationLoss, trainLoss));
        row.put("validationLossAboveTrainLoss", validationLoss > trainLoss);
        Double previousGap = previousRow == null ? null : generalizationGap(previousRow);
        if (previousGap == null) {
            row.remove("generalizationGapDelta");
            row.remove("generalizationGapTrend");
            row.remove("generalizationGapIncreasing");
            return;
        }
        putGapTrend(row, "generalizationGap", validationLoss - trainLoss - previousGap);
    }

    private static void putGapTrend(Map<String, Object> target, String prefix, double delta) {
        target.put(prefix + "Delta", delta);
        target.put(prefix + "Trend", trend(delta));
        target.put(prefix + "Increasing", delta > TREND_EPSILON);
    }

    private static String trend(double delta) {
        if (delta > TREND_EPSILON) {
            return "increasing";
        }
        if (delta < -TREND_EPSILON) {
            return "decreasing";
        }
        return "flat";
    }

    private static Double generalizationGap(Map<String, Object> row) {
        Double gap = finiteNumber(row.get("generalizationGap"));
        if (gap != null) {
            return gap;
        }
        Double trainLoss = finiteNumber(row.get("trainLoss"));
        Double validationLoss = finiteNumber(row.get("validationLoss"));
        return trainLoss == null || validationLoss == null ? null : validationLoss - trainLoss;
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }

    private static boolean isFinite(Double value) {
        return value != null && Double.isFinite(value);
    }
}
