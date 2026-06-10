package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Read-side view of persisted checkpoint manifest metadata.
 */
public record DiscreteTokenDatasetCheckpointManifestSnapshot(
        String schemaVersion,
        String experimentName,
        String runId,
        String modelFamily,
        long seed,
        long checkpointStep,
        long createdAtEpochMillis,
        String createdBy,
        DiscreteTokenDatasetFingerprint fingerprint,
        boolean datasetAccepted,
        String datasetGateStatus,
        Map<String, Object> datasetPlanReportMetadata,
        DiscreteTokenDatasetCheckpointLineage lineage,
        Map<String, Object> attributes) {

    public static final String SCHEMA_VERSION = DiscreteTokenDatasetCheckpointManifest.SCHEMA_VERSION;
    public static final String DATASET_PLAN_REPORT_METADATA_KEY =
            DiscreteTokenDatasetCheckpointManifest.DATASET_PLAN_REPORT_METADATA_KEY;

    public DiscreteTokenDatasetCheckpointManifestSnapshot {
        schemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(schemaVersion, "schemaVersion");
        experimentName = DiscreteTokenDatasetMetadataSupport.requireText(experimentName, "experimentName");
        runId = DiscreteTokenDatasetMetadataSupport.requireText(runId, "runId");
        modelFamily = DiscreteTokenDatasetMetadataSupport.requireText(modelFamily, "modelFamily");
        if (checkpointStep < 0L) {
            throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
        }
        if (createdAtEpochMillis < 0L) {
            throw new IllegalArgumentException("createdAtEpochMillis must be >= 0 but was " + createdAtEpochMillis);
        }
        createdBy = DiscreteTokenDatasetMetadataSupport.requireText(createdBy, "createdBy");
        fingerprint = Objects.requireNonNull(fingerprint, "fingerprint must not be null");
        datasetGateStatus = DiscreteTokenDatasetMetadataSupport.requireText(datasetGateStatus, "datasetGateStatus");
        datasetPlanReportMetadata = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(
                datasetPlanReportMetadata,
                "datasetPlanReportMetadata");
        lineage = lineage == null ? DiscreteTokenDatasetCheckpointLineage.root(runId) : lineage;
        attributes = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(attributes, "attributes");
        verifyDatasetReportConsistency(fingerprint, datasetAccepted, datasetGateStatus, datasetPlanReportMetadata);
    }

    public static DiscreteTokenDatasetCheckpointManifestSnapshot fromManifest(
            DiscreteTokenDatasetCheckpointManifest manifest) {
        Objects.requireNonNull(manifest, "manifest must not be null");
        return fromMetadata(manifest.toMetadata());
    }

    public static DiscreteTokenDatasetCheckpointManifestSnapshot fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointManifestSnapshot(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "schemaVersion"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "experimentName"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "runId"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "modelFamily"),
                DiscreteTokenDatasetMetadataSupport.requiredLong(metadata, "seed"),
                DiscreteTokenDatasetMetadataSupport.requiredLong(metadata, "checkpointStep"),
                DiscreteTokenDatasetMetadataSupport.requiredLong(metadata, "createdAtEpochMillis"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "createdBy"),
                DiscreteTokenDatasetFingerprint.fromMetadataSection(metadata),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "datasetAccepted"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "datasetGateStatus"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, DATASET_PLAN_REPORT_METADATA_KEY),
                DiscreteTokenDatasetCheckpointLineage.fromMetadata(
                        DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "lineage"),
                        DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "runId")),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "attributes"));
    }

    public boolean isCurrentSchema() {
        return SCHEMA_VERSION.equals(schemaVersion);
    }

    public DiscreteTokenDatasetFingerprint datasetPlanReportFingerprint() {
        return DiscreteTokenDatasetFingerprint.fromMetadataSection(datasetPlanReportMetadata);
    }

    public void requireCurrentSchema() {
        if (!isCurrentSchema()) {
            throw new IllegalStateException(
                    "checkpoint manifest schema is " + schemaVersion + " but expected " + SCHEMA_VERSION);
        }
    }

    public void requireDatasetAccepted() {
        if (!datasetAccepted) {
            throw new IllegalStateException("checkpoint dataset was not accepted: " + datasetGateStatus);
        }
    }

    public DiscreteTokenDatasetFingerprintMatch verifyPlan(DiscreteTokenDatasetPlan plan) {
        return DiscreteTokenDatasetFingerprintMatch.verify(fingerprint, plan);
    }

    public DiscreteTokenDatasetFingerprintMatch verifyReport(DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetFingerprintMatch.verify(fingerprint, report);
    }

    public DiscreteTokenDatasetCheckpointResumeReport resumeReport(DiscreteTokenDatasetPlan plan) {
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(this, plan);
    }

    public DiscreteTokenDatasetCheckpointResumeReport resumeReport(
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumePolicy policy) {
        Objects.requireNonNull(policy, "policy must not be null");
        return policy.evaluate(this, plan);
    }

    public DiscreteTokenDatasetCheckpointResumeReport resumeReport(
            DiscreteTokenDatasetPlan plan,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(this, plan, expectation);
    }

    public DiscreteTokenDatasetCheckpointResumeReport resumeReport(DiscreteTokenDatasetPlanReport report) {
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(this, report);
    }

    public DiscreteTokenDatasetCheckpointResumeReport resumeReport(
            DiscreteTokenDatasetPlanReport report,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation) {
        return DiscreteTokenDatasetCheckpointResumeReport.fromSnapshot(this, report, expectation);
    }

    public String summary() {
        return "checkpoint "
                + runId
                + " step "
                + checkpointStep
                + " dataset "
                + fingerprint.shortValue()
                + " "
                + datasetGateStatus;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("schemaVersion", schemaVersion);
        metadata.put("experimentName", experimentName);
        metadata.put("runId", runId);
        metadata.put("modelFamily", modelFamily);
        metadata.put("seed", seed);
        metadata.put("checkpointStep", checkpointStep);
        metadata.put("createdAtEpochMillis", createdAtEpochMillis);
        metadata.put("createdBy", createdBy);
        metadata.put("fingerprint", fingerprint.toMetadata());
        metadata.put("datasetAccepted", datasetAccepted);
        metadata.put("datasetGateStatus", datasetGateStatus);
        metadata.put(DATASET_PLAN_REPORT_METADATA_KEY, datasetPlanReportMetadata);
        metadata.put("lineage", lineage.toMetadata());
        metadata.put("attributes", attributes);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyDatasetReportConsistency(
            DiscreteTokenDatasetFingerprint fingerprint,
            boolean datasetAccepted,
            String datasetGateStatus,
            Map<String, Object> datasetPlanReportMetadata) {
        DiscreteTokenDatasetFingerprint reportFingerprint =
                DiscreteTokenDatasetFingerprint.fromMetadataSection(datasetPlanReportMetadata);
        if (!fingerprint.equals(reportFingerprint)) {
            throw new IllegalArgumentException(
                    "top-level fingerprint must match datasetPlanReport fingerprint");
        }

        boolean reportAccepted = DiscreteTokenDatasetMetadataSupport.requiredBoolean(
                datasetPlanReportMetadata,
                "accepted");
        if (datasetAccepted != reportAccepted) {
            throw new IllegalArgumentException(
                    "top-level datasetAccepted must match datasetPlanReport accepted");
        }

        String reportGateStatus = DiscreteTokenDatasetMetadataSupport.requiredString(
                datasetPlanReportMetadata,
                "gateStatus");
        if (!datasetGateStatus.equals(reportGateStatus)) {
            throw new IllegalArgumentException(
                    "top-level datasetGateStatus must match datasetPlanReport gateStatus");
        }
    }

}
