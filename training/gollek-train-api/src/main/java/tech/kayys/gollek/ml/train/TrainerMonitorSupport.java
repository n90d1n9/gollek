package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Monitor metric normalization and lookup helpers.
 */
final class TrainerMonitorSupport {
    private TrainerMonitorSupport() {
    }

    static String normalizeMetric(String metricName) {
        return normalizeMetric(metricName, "monitor metric");
    }

    static String normalizeMetric(String metricName, String label) {
        if (metricName == null || metricName.isBlank()) {
            return null;
        }
        String normalized = metricName.trim();
        if (normalized.startsWith("validationMetric.")) {
            normalized = normalized.substring("validationMetric.".length());
        }
        if (normalized.isBlank()) {
            throw new IllegalArgumentException(label + " must not be blank");
        }
        return TrainerMetricSnapshots.requireName(normalized);
    }

    static void requireMetricPresent(String metricName, List<TrainingMetric> validationMetrics) {
        requireMetricPresent(metricName, validationMetrics, "monitor metric");
    }

    static void requireMetricPresent(String metricName, List<TrainingMetric> validationMetrics, String label) {
        if (metricName == null) {
            return;
        }
        for (TrainingMetric metric : validationMetrics) {
            if (metricName.equals(TrainerMetricSnapshots.requireName(metric.name()))) {
                return;
            }
        }
        throw new IllegalArgumentException(label + " '" + metricName
                + "' is not registered. Add the matching metric before build().");
    }

    static String label(String metricName) {
        return metricName == null ? "validation_loss" : "validationMetric." + metricName;
    }

    static double valueOrThrow(
            String metricName,
            double validationLoss,
            Map<String, Double> validationMetrics,
            String label) {
        if (metricName == null) {
            return validationLoss;
        }
        Double metricValue = validationMetrics.get(metricName);
        if (metricValue == null) {
            throw new IllegalStateException(label + " '" + metricName
                    + "' is not available. Add it with .metric(...) or trainingOptions().*Metric().");
        }
        return metricValue;
    }

    static double valueOrNaN(
            String metricName,
            double validationLoss,
            Map<String, Double> validationMetrics) {
        if (metricName == null) {
            return validationLoss;
        }
        Double metricValue = validationMetrics.get(metricName);
        return metricValue == null ? Double.NaN : metricValue;
    }

    static int earlyStoppingBestEpoch(String metricName, TrainingSummary base, int customBestEpoch) {
        return metricName == null ? base.bestValidationEpoch() : customBestEpoch;
    }

    static double earlyStoppingBestValue(String metricName, TrainingSummary base, double customBestValue) {
        return metricName == null ? base.bestValidationLoss() : customBestValue;
    }

    static double earlyStoppingLatestValue(String metricName, TrainingSummary base, double customLatestValue) {
        return metricName == null
                ? (base.latestValidationLoss() == null ? Double.NaN : base.latestValidationLoss())
                : customLatestValue;
    }

    static int earlyStoppingEpochsWithoutImprovement(
            String metricName,
            TrainingSummary base,
            int customEpochsWithoutImprovement) {
        if (metricName != null) {
            return customEpochsWithoutImprovement;
        }
        Object value = base.metadata().get("epochsWithoutImprovement");
        return value instanceof Number number ? number.intValue() : 0;
    }
}
