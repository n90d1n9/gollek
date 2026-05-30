package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Publishes progress against the best validation checkpoint.
 */
final class TrainerValidationProgressMetadata {
    private TrainerValidationProgressMetadata() {
    }

    static void put(
            Map<String, Object> metadata,
            int epochCount,
            double bestValidationLoss,
            int bestValidationEpoch,
            Double latestValidationLoss) {
        boolean available = bestValidationEpoch >= 0
                && Double.isFinite(bestValidationLoss)
                && latestValidationLoss != null
                && Double.isFinite(latestValidationLoss);
        metadata.put("validationProgressAvailable", available);
        if (!available) {
            return;
        }
        metadata.put("latestValidationLossDeltaFromBest", latestValidationLoss - bestValidationLoss);
        metadata.put(
                "latestValidationLossRatioToBest",
                TrainerOptimizationMetadata.ratio(latestValidationLoss, bestValidationLoss));
        metadata.put("epochsSinceBestValidation", Math.max(0, epochCount - 1 - bestValidationEpoch));
        metadata.put("latestValidationLossIsBest", latestValidationLoss.doubleValue() == bestValidationLoss);
    }

    static void putEpoch(Map<String, Object> row, List<Map<String, Object>> rows) {
        Progress progress = progressThroughEpoch(row, rows);
        row.put("validationLossProgressAvailable", progress.available());
        if (!progress.available()) {
            row.remove("validationLossDeltaFromBest");
            row.remove("validationLossRatioToBest");
            row.remove("epochsSinceBestValidation");
            row.remove("validationLossIsBest");
            return;
        }
        row.put("validationLossDeltaFromBest", progress.currentLoss() - progress.bestLoss());
        row.put("validationLossRatioToBest", TrainerOptimizationMetadata.ratio(progress.currentLoss(), progress.bestLoss()));
        row.put("epochsSinceBestValidation", Math.max(0, progress.currentEpoch() - progress.bestEpoch()));
        row.put("validationLossIsBest", progress.currentLoss() == progress.bestLoss());
    }

    private static Progress progressThroughEpoch(Map<String, Object> currentRow, List<Map<String, Object>> rows) {
        Double currentLoss = finiteNumber(currentRow.get("validationLoss"));
        if (currentLoss == null) {
            return Progress.unavailable();
        }
        int currentEpoch = epoch(currentRow.get("epoch"), Integer.MAX_VALUE);
        double bestLoss = Double.POSITIVE_INFINITY;
        int bestEpoch = -1;
        boolean available = false;
        for (Map<String, Object> row : rows) {
            int candidateEpoch = epoch(row.get("epoch"), -1);
            if (candidateEpoch < 0 || candidateEpoch > currentEpoch) {
                continue;
            }
            Double loss = finiteNumber(row.get("validationLoss"));
            if (loss == null) {
                continue;
            }
            if (!available || loss < bestLoss) {
                bestLoss = loss;
                bestEpoch = candidateEpoch;
                available = true;
            }
        }
        if (!rows.contains(currentRow)) {
            if (!available || currentLoss < bestLoss) {
                bestLoss = currentLoss;
                bestEpoch = currentEpoch;
                available = true;
            }
        }
        return available
                ? new Progress(true, currentEpoch, currentLoss, bestEpoch, bestLoss)
                : Progress.unavailable();
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
            double currentLoss,
            int bestEpoch,
            double bestLoss) {
        static Progress unavailable() {
            return new Progress(false, -1, Double.NaN, -1, Double.NaN);
        }
    }
}
