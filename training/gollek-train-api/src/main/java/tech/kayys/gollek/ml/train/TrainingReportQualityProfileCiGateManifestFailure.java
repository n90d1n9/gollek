package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Typed manifest verification failure for CI dashboards and release gates.
 */
public record TrainingReportQualityProfileCiGateManifestFailure(
        TrainingReportQualityProfileCiGateManifestFailureCategory category,
        String message) {
    public TrainingReportQualityProfileCiGateManifestFailure {
        category = Objects.requireNonNull(category, "category must not be null");
        if (message == null || message.isBlank()) {
            throw new IllegalArgumentException("message must not be blank");
        }
        message = message.trim();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("category", category.id());
        map.put("message", message);
        return Map.copyOf(map);
    }
}
