package tech.kayys.gollek.ml.train;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Publishes latest metric snapshots into the stable training-summary metadata
 * shape consumed by reports, examples, and SDK callers.
 */
final class TrainerMetricsMetadata {
    private TrainerMetricsMetadata() {
    }

    static void putLatest(
            Map<String, Object> metadata,
            boolean metricsEnabled,
            Map<String, Double> latestTrainMetrics,
            Map<String, Double> latestValidationMetrics,
            Map<String, Object> latestTrainMetricDetails,
            Map<String, Object> latestValidationMetricDetails) {
        Map<String, Double> trainMetrics = doubleSnapshot(latestTrainMetrics);
        Map<String, Double> validationMetrics = doubleSnapshot(latestValidationMetrics);
        Map<String, Object> trainDetails = objectSnapshot(latestTrainMetricDetails);
        Map<String, Object> validationDetails = objectSnapshot(latestValidationMetricDetails);

        metadata.put("metricsEnabled", metricsEnabled);
        metadata.put("latestTrainMetrics", trainMetrics);
        metadata.put("latestValidationMetrics", validationMetrics);
        metadata.put("latestTrainMetricDetails", trainDetails);
        metadata.put("latestValidationMetricDetails", validationDetails);
        TrainerMetadataSupport.flatten(metadata, "trainMetric.", trainMetrics);
        TrainerMetadataSupport.flatten(metadata, "validationMetric.", validationMetrics);
        TrainerMetadataSupport.flatten(metadata, "trainMetricDetails.", trainDetails);
        TrainerMetadataSupport.flatten(metadata, "validationMetricDetails.", validationDetails);
    }

    private static Map<String, Double> doubleSnapshot(Map<String, Double> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> objectSnapshot(Map<String, Object> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(source);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
