package tech.kayys.gollek.ml.train;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Owns the latest trainer metric snapshots published by the runtime.
 */
final class TrainerMetricState {
    private volatile Map<String, Double> trainValues = Map.of();
    private volatile Map<String, Double> validationValues = Map.of();
    private volatile Map<String, Object> trainDetails = Map.of();
    private volatile Map<String, Object> validationDetails = Map.of();

    void reset() {
        trainValues = Map.of();
        validationValues = Map.of();
        trainDetails = Map.of();
        validationDetails = Map.of();
    }

    void recordTrain(TrainerMetricRuntime.MetricSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        trainValues = immutableCopy(snapshot.values());
        trainDetails = immutableCopy(snapshot.details());
    }

    void recordValidation(TrainerMetricRuntime.MetricSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        validationValues = immutableCopy(snapshot.values());
        validationDetails = immutableCopy(snapshot.details());
    }

    Map<String, Double> trainValues() {
        return trainValues;
    }

    Map<String, Double> validationValues() {
        return validationValues;
    }

    Map<String, Object> trainDetails() {
        return trainDetails;
    }

    Map<String, Object> validationDetails() {
        return validationDetails;
    }

    private static <T> Map<String, T> immutableCopy(Map<String, T> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }
}
