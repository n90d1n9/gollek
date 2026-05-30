package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Result of applying a dataset-plan readiness gate.
 */
public record DiscreteTokenDatasetPlanReadinessReport(
        DiscreteTokenDatasetPlanDiagnostics diagnostics,
        boolean failOnWarnings) {

    public DiscreteTokenDatasetPlanReadinessReport {
        diagnostics = Objects.requireNonNull(diagnostics, "diagnostics must not be null");
    }

    public static DiscreteTokenDatasetPlanReadinessReport fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        DiscreteTokenDatasetPlanReadinessReport report = new DiscreteTokenDatasetPlanReadinessReport(
                DiscreteTokenDatasetPlanDiagnostics.fromMetadata(requiredMap(metadata, "diagnostics")),
                requiredBoolean(metadata, "failOnWarnings"));
        verifyOptionalMetadata(metadata, report);
        return report;
    }

    public boolean accepted() {
        return diagnostics.isReadyForTraining() && !blockedByWarnings();
    }

    public boolean blockedByWarnings() {
        return failOnWarnings && diagnostics.hasWarnings();
    }

    public String gateStatus() {
        if (!diagnostics.isReadyForTraining()) {
            return "blocked";
        }
        if (blockedByWarnings()) {
            return "warning-blocked";
        }
        return "accepted";
    }

    public List<String> rejectionReasons() {
        if (accepted()) {
            return List.of();
        }

        List<String> reasons = new ArrayList<>();
        if (!diagnostics.isReadyForTraining()) {
            reasons.add("dataset is not ready for training");
        }
        if (!diagnostics.warnings().isEmpty()) {
            reasons.addAll(diagnostics.warnings());
        }
        return List.copyOf(reasons);
    }

    public String summary() {
        if (accepted()) {
            return "dataset plan accepted";
        }
        return "dataset plan " + gateStatus() + ": " + String.join("; ", rejectionReasons());
    }

    public void requireAccepted() {
        if (!accepted()) {
            throw new IllegalStateException(summary());
        }
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("gateStatus", gateStatus());
        metadata.put("accepted", accepted());
        metadata.put("failOnWarnings", failOnWarnings);
        metadata.put("blockedByWarnings", blockedByWarnings());
        metadata.put("rejectionReasons", rejectionReasons());
        metadata.put("diagnostics", diagnostics.toMetadata());
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

    private static String optionalString(Map<?, ?> metadata, String key, String expected) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return expected;
        }
        Object value = metadata.get(key);
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static boolean optionalBoolean(Map<?, ?> metadata, String key, boolean expected) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return expected;
        }
        Object value = metadata.get(key);
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

    private static void verifyOptionalMetadata(
            Map<?, ?> metadata,
            DiscreteTokenDatasetPlanReadinessReport report) {
        if (!optionalString(metadata, "gateStatus", report.gateStatus()).equals(report.gateStatus())) {
            throw new IllegalArgumentException("metadata field 'gateStatus' does not match readiness report");
        }
        if (optionalBoolean(metadata, "accepted", report.accepted()) != report.accepted()) {
            throw new IllegalArgumentException("metadata field 'accepted' does not match readiness report");
        }
        if (optionalBoolean(metadata, "blockedByWarnings", report.blockedByWarnings()) != report.blockedByWarnings()) {
            throw new IllegalArgumentException(
                    "metadata field 'blockedByWarnings' does not match readiness report");
        }
    }
}
