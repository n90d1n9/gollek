package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One-stop preflight report for a materialized token dataset plan.
 */
public record DiscreteTokenDatasetPlanReport(
        DiscreteTokenDatasetFingerprint fingerprint,
        DiscreteTokenDatasetPlanReadinessReport readiness) {

    public DiscreteTokenDatasetPlanReport {
        fingerprint = Objects.requireNonNull(fingerprint, "fingerprint must not be null");
        readiness = Objects.requireNonNull(readiness, "readiness must not be null");
        if (fingerprint.exampleCount() != readiness.diagnostics().exampleCount()) {
            throw new IllegalArgumentException(
                    "fingerprint example count must match diagnostics example count");
        }
    }

    public static DiscreteTokenDatasetPlanReport from(DiscreteTokenDatasetPlan plan) {
        return from(plan, DiscreteTokenDatasetPlanReadinessGate.training());
    }

    public static DiscreteTokenDatasetPlanReport from(
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetPlanReadinessGate gate) {
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(gate, "gate must not be null");
        return new DiscreteTokenDatasetPlanReport(
                plan.fingerprint(),
                gate.evaluate(plan));
    }

    public static DiscreteTokenDatasetPlanReport fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        DiscreteTokenDatasetPlanReport report = new DiscreteTokenDatasetPlanReport(
                DiscreteTokenDatasetFingerprint.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "fingerprint")),
                DiscreteTokenDatasetPlanReadinessReport.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "readiness")));
        verifyOptionalMetadata(metadata, report);
        return report;
    }

    public DiscreteTokenDatasetPlanDiagnostics diagnostics() {
        return readiness.diagnostics();
    }

    public boolean accepted() {
        return readiness.accepted();
    }

    public String gateStatus() {
        return readiness.gateStatus();
    }

    public List<String> warnings() {
        return diagnostics().warnings();
    }

    public void requireAccepted() {
        readiness.requireAccepted();
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprint(
            DiscreteTokenDatasetFingerprint expectedFingerprint) {
        return DiscreteTokenDatasetFingerprintMatch.verify(expectedFingerprint, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadata(Map<?, ?> expectedFingerprintMetadata) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadata(expectedFingerprintMetadata, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadataSection(Map<?, ?> expectedMetadata) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(expectedMetadata, this);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyFingerprintMetadataSection(
            Map<?, ?> expectedMetadata,
            String key) {
        return DiscreteTokenDatasetFingerprintMatch.verifyMetadataSection(expectedMetadata, key, this);
    }

    public String summary() {
        String prefix = "dataset plan " + fingerprint.shortValue() + " ";
        if (accepted()) {
            return prefix + "accepted";
        }
        return prefix + gateStatus() + ": " + String.join("; ", readiness.rejectionReasons());
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("fingerprint", fingerprint.toMetadata());
        metadata.put("accepted", accepted());
        metadata.put("gateStatus", gateStatus());
        metadata.put("readiness", readiness.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyOptionalMetadata(
            Map<?, ?> metadata,
            DiscreteTokenDatasetPlanReport report) {
        if (DiscreteTokenDatasetMetadataSupport.optionalBoolean(metadata, "accepted", report.accepted())
                != report.accepted()) {
            throw new IllegalArgumentException("metadata field 'accepted' does not match plan report");
        }
        if (!DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "gateStatus", report.gateStatus())
                .equals(report.gateStatus())) {
            throw new IllegalArgumentException("metadata field 'gateStatus' does not match plan report");
        }
    }
}
