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
                DiscreteTokenDatasetPlanDiagnostics.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "diagnostics")),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "failOnWarnings"));
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

    private static void verifyOptionalMetadata(
            Map<?, ?> metadata,
            DiscreteTokenDatasetPlanReadinessReport report) {
        if (!DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "gateStatus", report.gateStatus())
                .equals(report.gateStatus())) {
            throw new IllegalArgumentException("metadata field 'gateStatus' does not match readiness report");
        }
        if (DiscreteTokenDatasetMetadataSupport.optionalBoolean(metadata, "accepted", report.accepted())
                != report.accepted()) {
            throw new IllegalArgumentException("metadata field 'accepted' does not match readiness report");
        }
        if (DiscreteTokenDatasetMetadataSupport.optionalBoolean(
                        metadata,
                        "blockedByWarnings",
                        report.blockedByWarnings())
                != report.blockedByWarnings()) {
            throw new IllegalArgumentException(
                    "metadata field 'blockedByWarnings' does not match readiness report");
        }
    }
}
