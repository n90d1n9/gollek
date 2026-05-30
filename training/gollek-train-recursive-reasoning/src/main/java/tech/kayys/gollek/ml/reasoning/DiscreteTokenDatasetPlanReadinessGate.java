package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Converts dataset-plan diagnostics into a trainer or CI readiness decision.
 */
public record DiscreteTokenDatasetPlanReadinessGate(
        DiscreteTokenDatasetPlanDiagnosticsPolicy diagnosticsPolicy,
        boolean failOnWarnings) {

    public DiscreteTokenDatasetPlanReadinessGate {
        diagnosticsPolicy = Objects.requireNonNull(diagnosticsPolicy, "diagnosticsPolicy must not be null");
    }

    public static DiscreteTokenDatasetPlanReadinessGate training() {
        return new DiscreteTokenDatasetPlanReadinessGate(
                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults(),
                false);
    }

    public static DiscreteTokenDatasetPlanReadinessGate strict() {
        return new DiscreteTokenDatasetPlanReadinessGate(
                DiscreteTokenDatasetPlanDiagnosticsPolicy.defaults(),
                true);
    }

    public static DiscreteTokenDatasetPlanReadinessGate lenient() {
        return new DiscreteTokenDatasetPlanReadinessGate(
                DiscreteTokenDatasetPlanDiagnosticsPolicy.lenient(),
                false);
    }

    public static DiscreteTokenDatasetPlanReadinessGate fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetPlanReadinessGate(
                DiscreteTokenDatasetPlanDiagnosticsPolicy.fromMetadata(requiredMap(metadata, "diagnosticsPolicy")),
                requiredBoolean(metadata, "failOnWarnings"));
    }

    public DiscreteTokenDatasetPlanReadinessReport evaluate(DiscreteTokenDatasetPlan plan) {
        return evaluate(DiscreteTokenDatasetPlanDiagnostics.from(plan, diagnosticsPolicy));
    }

    public DiscreteTokenDatasetPlanReadinessReport evaluate(DiscreteTokenDatasetPlanDiagnostics diagnostics) {
        return new DiscreteTokenDatasetPlanReadinessReport(diagnostics, failOnWarnings);
    }

    public DiscreteTokenDatasetPlanDiagnostics requireReady(DiscreteTokenDatasetPlan plan) {
        DiscreteTokenDatasetPlanReadinessReport report = evaluate(plan);
        report.requireAccepted();
        return report.diagnostics();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("failOnWarnings", failOnWarnings);
        metadata.put("diagnosticsPolicy", diagnosticsPolicy.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Map<?, ?> requiredMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static boolean requiredBoolean(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof CharSequence text) {
            String normalized = text.toString().trim().toLowerCase();
            if ("true".equals(normalized)) {
                return true;
            }
            if ("false".equals(normalized)) {
                return false;
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a boolean");
    }

    private static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }
}
