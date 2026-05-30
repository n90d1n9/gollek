package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resume preflight for a persisted checkpoint manifest and a current dataset plan.
 */
public record DiscreteTokenDatasetCheckpointResumeReport(
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
        DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation,
        DiscreteTokenDatasetPlanReport currentPlanReport,
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
        Map<String, Object> policyMetadata) {

    public DiscreteTokenDatasetCheckpointResumeReport {
        checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        fingerprintMatch = Objects.requireNonNull(fingerprintMatch, "fingerprintMatch must not be null");
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
        compatibilityMode = Objects.requireNonNull(compatibilityMode, "compatibilityMode must not be null");
        policyMetadata = immutableMetadataMap(policyMetadata, "policyMetadata");
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport) {
        this(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT,
                Map.of());
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            Map<String, Object> policyMetadata) {
        this(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT,
                policyMetadata);
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetPlanReport currentPlanReport,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {
        this(checkpoint, fingerprintMatch, expectation, currentPlanReport, compatibilityMode, Map.of());
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        this(checkpoint, fingerprintMatch, expectation, null, DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public DiscreteTokenDatasetCheckpointResumeReport(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch) {
        this(checkpoint, fingerprintMatch, DiscreteTokenDatasetCheckpointResumeExpectation.none(), null);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan) {
        return fromMetadata(checkpointMetadata, plan, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(checkpointMetadata),
                plan,
                expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlanReport report) {
        return fromMetadata(checkpointMetadata, report, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromMetadata(
            Map<?, ?> checkpointMetadata,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(checkpointMetadata),
                report,
                expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan) {
        return fromSnapshot(checkpoint, plan, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        Objects.requireNonNull(plan, "plan must not be null");
        return fromSnapshot(checkpoint, plan.report(), expectation);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report) {
        return fromSnapshot(checkpoint, report, DiscreteTokenDatasetCheckpointResumeExpectation.none());
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return fromSnapshot(
                checkpoint,
                report,
                expectation,
                DiscreteTokenDatasetCheckpointResumeCompatibilityMode.STRICT);
    }

    public static DiscreteTokenDatasetCheckpointResumeReport fromSnapshot(
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint,
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode) {
        Objects.requireNonNull(report, "report must not be null");
        return new DiscreteTokenDatasetCheckpointResumeReport(
                checkpoint,
                checkpoint.verifyReport(report),
                expectation,
                report,
                compatibilityMode);
    }

    public boolean schemaAccepted() {
        return checkpoint.isCurrentSchema();
    }

    public boolean datasetAccepted() {
        return checkpoint.datasetAccepted();
    }

    public boolean fingerprintMatched() {
        return fingerprintMatch.matches();
    }

    public boolean expectationAccepted() {
        return expectation.accepts(checkpoint);
    }

    public boolean currentPlanChecked() {
        return currentPlanReport != null;
    }

    public boolean policyTracked() {
        return !policyMetadata.isEmpty();
    }

    public boolean currentPlanAccepted() {
        return currentPlanReport == null || currentPlanReport.accepted();
    }

    public String currentPlanGateStatus() {
        return currentPlanReport == null ? "not-checked" : currentPlanReport.gateStatus();
    }

    public boolean ready() {
        return schemaAccepted()
                && (datasetAccepted() || compatibilityMode.allowRejectedCheckpointDataset())
                && (fingerprintMatched() || compatibilityMode.allowFingerprintMismatch())
                && expectationAccepted()
                && (currentPlanAccepted() || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus()));
    }

    public String status() {
        return ready() ? "ready" : "blocked";
    }

    public List<String> rejectionReasons() {
        if (ready()) {
            return List.of();
        }

        List<String> reasons = new ArrayList<>();
        if (!schemaAccepted()) {
            reasons.add("checkpoint manifest schema is "
                    + checkpoint.schemaVersion()
                    + " but expected "
                    + DiscreteTokenDatasetCheckpointManifestSnapshot.SCHEMA_VERSION);
        }
        if (!datasetAccepted() && !compatibilityMode.allowRejectedCheckpointDataset()) {
            reasons.add("checkpoint dataset was not accepted: " + checkpoint.datasetGateStatus());
        }
        if (!fingerprintMatched() && !compatibilityMode.allowFingerprintMismatch()) {
            reasons.add(fingerprintMatch.summary());
        }
        reasons.addAll(expectation.rejectionReasons(checkpoint));
        if (!currentPlanAccepted() && !compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus())) {
            reasons.add("current dataset plan was not accepted: " + currentPlanGateStatus());
        }
        return List.copyOf(reasons);
    }

    public boolean forceAccepted() {
        return compatibilityMode.force() && ready() && !compatibilityWarnings().isEmpty();
    }

    public List<String> compatibilityWarnings() {
        List<String> warnings = new ArrayList<>();
        if (!datasetAccepted() && compatibilityMode.allowRejectedCheckpointDataset()) {
            warnings.add("force mode accepted a checkpoint dataset that was not accepted: "
                    + checkpoint.datasetGateStatus());
        }
        if (!fingerprintMatched() && compatibilityMode.allowFingerprintMismatch()) {
            warnings.add("force mode accepted dataset fingerprint mismatch: " + fingerprintMatch.summary());
        }
        if (!currentPlanAccepted() && compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus())) {
            warnings.add(compatibilityMode.id()
                    + " mode accepted current dataset plan status: "
                    + currentPlanGateStatus());
        }
        return List.copyOf(warnings);
    }

    public String summary() {
        String prefix = "checkpoint resume "
                + status()
                + ": "
                + checkpoint.runId()
                + " step "
                + checkpoint.checkpointStep()
                + " dataset "
                + checkpoint.fingerprint().shortValue();
        if (ready()) {
            return prefix;
        }
        return prefix + " (" + String.join("; ", rejectionReasons()) + ")";
    }

    public void requireReady() {
        if (!ready()) {
            throw new IllegalStateException(summary());
        }
    }

    public DiscreteTokenDatasetCheckpointResumeExplanation explanation() {
        return DiscreteTokenDatasetCheckpointResumeExplanation.from(this);
    }

    public DiscreteTokenDatasetCheckpointResumeReport withPolicyMetadata(Map<String, Object> policyMetadata) {
        return new DiscreteTokenDatasetCheckpointResumeReport(
                checkpoint,
                fingerprintMatch,
                expectation,
                currentPlanReport,
                compatibilityMode,
                policyMetadata);
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status());
        metadata.put("ready", ready());
        metadata.put("schemaAccepted", schemaAccepted());
        metadata.put("datasetAccepted", datasetAccepted());
        metadata.put("fingerprintMatched", fingerprintMatched());
        metadata.put("expectationAccepted", expectationAccepted());
        metadata.put("currentPlanChecked", currentPlanChecked());
        metadata.put("currentPlanAccepted", currentPlanAccepted());
        metadata.put("currentPlanGateStatus", currentPlanGateStatus());
        metadata.put("compatibilityMode", compatibilityMode.id());
        metadata.put("forceAccepted", forceAccepted());
        metadata.put("compatibilityWarnings", compatibilityWarnings());
        metadata.put("policyTracked", policyTracked());
        metadata.put("runId", checkpoint.runId());
        metadata.put("checkpointStep", checkpoint.checkpointStep());
        metadata.put("schemaVersion", checkpoint.schemaVersion());
        metadata.put("datasetGateStatus", checkpoint.datasetGateStatus());
        metadata.put("rejectionReasons", rejectionReasons());
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("fingerprintMatch", fingerprintMatch.toMetadata());
        if (policyTracked()) {
            metadata.put("policy", policyMetadata);
        }
        if (currentPlanReport != null) {
            metadata.put("currentPlanReport", currentPlanReport.toMetadata());
        }
        metadata.put("checkpoint", checkpoint.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static Map<String, Object> immutableMetadataMap(Map<String, Object> metadata, String name) {
        if (metadata == null || metadata.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            String key = Objects.requireNonNull(entry.getKey(), name + " key must not be null");
            if (key.isBlank()) {
                throw new IllegalArgumentException(name + " key must not be blank");
            }
            copy.put(key, Objects.requireNonNull(entry.getValue(), name + " field '" + key + "' must not be null"));
        }
        return Collections.unmodifiableMap(copy);
    }
}
