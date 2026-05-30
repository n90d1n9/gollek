package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

/**
 * Publishes best-loss and non-improvement streak diagnostics from epoch history.
 */
final class TrainerLossImprovementMetadata {
    private TrainerLossImprovementMetadata() {
    }

    static void putLatest(Map<String, Object> metadata, List<Map<String, Object>> rows) {
        putLatest(metadata, rows, "trainLoss", "latestTrainLoss");
        putLatest(metadata, rows, "validationLoss", "latestValidationLoss");
    }

    static void putEpoch(Map<String, Object> row, List<Map<String, Object>> rows, String lossKey) {
        String bestKey = lossKey + "Best";
        String bestEpochKey = lossKey + "BestEpoch";
        String streakKey = lossKey + "NonImprovingStreak";
        String bestAtEpochKey = lossKey + "BestAtEpoch";
        BestLoss best = bestLossThroughEpoch(row, rows, lossKey);
        if (!best.available()) {
            row.remove(bestKey);
            row.remove(bestEpochKey);
            row.remove(streakKey);
            row.remove(bestAtEpochKey);
            return;
        }
        row.put(bestKey, best.loss());
        row.put(bestEpochKey, best.epoch());
        row.put(streakKey, best.nonImprovingStreak());
        row.put(bestAtEpochKey, best.nonImprovingStreak() == 0);
    }

    private static void putLatest(
            Map<String, Object> metadata,
            List<Map<String, Object>> rows,
            String lossKey,
            String prefix) {
        String availabilityKey = prefix + "BestAvailable";
        String bestKey = prefix + "Best";
        String bestEpochKey = prefix + "BestEpoch";
        String streakKey = prefix + "NonImprovingStreak";
        String bestAtLatestKey = prefix + "BestAtLatestEpoch";
        BestLoss best = bestLoss(rows, lossKey);
        metadata.put(availabilityKey, best.available());
        if (!best.available()) {
            metadata.remove(bestKey);
            metadata.remove(bestEpochKey);
            metadata.remove(streakKey);
            metadata.remove(bestAtLatestKey);
            return;
        }
        metadata.put(bestKey, best.loss());
        metadata.put(bestEpochKey, best.epoch());
        metadata.put(streakKey, best.nonImprovingStreak());
        metadata.put(bestAtLatestKey, best.nonImprovingStreak() == 0);
    }

    private static BestLoss bestLoss(List<Map<String, Object>> rows, String lossKey) {
        double best = Double.POSITIVE_INFINITY;
        int bestEpoch = -1;
        int nonImprovingStreak = 0;
        int observationIndex = 0;
        boolean available = false;
        for (Map<String, Object> row : rows) {
            Double loss = finiteNumber(row.get(lossKey));
            if (loss == null) {
                continue;
            }
            int epoch = epoch(row.get("epoch"), observationIndex);
            if (!available || loss < best) {
                best = loss;
                bestEpoch = epoch;
                nonImprovingStreak = 0;
                available = true;
            } else {
                nonImprovingStreak++;
            }
            observationIndex++;
        }
        return new BestLoss(available, best, bestEpoch, nonImprovingStreak);
    }

    private static BestLoss bestLossThroughEpoch(
            Map<String, Object> currentRow,
            List<Map<String, Object>> rows,
            String lossKey) {
        Double currentLoss = finiteNumber(currentRow.get(lossKey));
        if (currentLoss == null) {
            return new BestLoss(false, Double.NaN, -1, 0);
        }
        int currentEpoch = epoch(currentRow.get("epoch"), Integer.MAX_VALUE);
        double best = Double.POSITIVE_INFINITY;
        int bestEpoch = -1;
        int nonImprovingStreak = 0;
        boolean available = false;
        for (Map<String, Object> row : rows) {
            int candidateEpoch = epoch(row.get("epoch"), -1);
            if (candidateEpoch < 0 || candidateEpoch > currentEpoch || !row.containsKey(lossKey)) {
                continue;
            }
            Double loss = finiteNumber(row.get(lossKey));
            if (loss == null) {
                continue;
            }
            if (!available || loss < best) {
                best = loss;
                bestEpoch = candidateEpoch;
                nonImprovingStreak = 0;
                available = true;
            } else {
                nonImprovingStreak++;
            }
        }
        if (!rows.contains(currentRow)) {
            if (!available || currentLoss < best) {
                best = currentLoss;
                bestEpoch = currentEpoch;
                nonImprovingStreak = 0;
                available = true;
            } else {
                nonImprovingStreak++;
            }
        }
        return new BestLoss(available, best, bestEpoch, nonImprovingStreak);
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

    private record BestLoss(boolean available, double loss, int epoch, int nonImprovingStreak) {
    }
}
