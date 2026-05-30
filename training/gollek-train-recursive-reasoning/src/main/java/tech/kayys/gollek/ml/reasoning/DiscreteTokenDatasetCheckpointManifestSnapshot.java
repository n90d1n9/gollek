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
        schemaVersion = requireText(schemaVersion, "schemaVersion");
        experimentName = requireText(experimentName, "experimentName");
        runId = requireText(runId, "runId");
        modelFamily = requireText(modelFamily, "modelFamily");
        if (checkpointStep < 0L) {
            throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
        }
        if (createdAtEpochMillis < 0L) {
            throw new IllegalArgumentException("createdAtEpochMillis must be >= 0 but was " + createdAtEpochMillis);
        }
        createdBy = requireText(createdBy, "createdBy");
        fingerprint = Objects.requireNonNull(fingerprint, "fingerprint must not be null");
        datasetGateStatus = requireText(datasetGateStatus, "datasetGateStatus");
        datasetPlanReportMetadata = immutableMetadataMap(
                datasetPlanReportMetadata,
                "datasetPlanReportMetadata");
        lineage = lineage == null ? DiscreteTokenDatasetCheckpointLineage.root(runId) : lineage;
        attributes = immutableMetadataMap(attributes, "attributes");
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
                requiredString(metadata, "schemaVersion"),
                requiredString(metadata, "experimentName"),
                requiredString(metadata, "runId"),
                requiredString(metadata, "modelFamily"),
                requiredLong(metadata, "seed"),
                requiredLong(metadata, "checkpointStep"),
                requiredLong(metadata, "createdAtEpochMillis"),
                requiredString(metadata, "createdBy"),
                DiscreteTokenDatasetFingerprint.fromMetadataSection(metadata),
                requiredBoolean(metadata, "datasetAccepted"),
                requiredString(metadata, "datasetGateStatus"),
                requiredMap(metadata, DATASET_PLAN_REPORT_METADATA_KEY),
                DiscreteTokenDatasetCheckpointLineage.fromMetadata(optionalMap(metadata, "lineage"),
                        requiredString(metadata, "runId")),
                optionalMap(metadata, "attributes"));
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

        boolean reportAccepted = requiredBoolean(datasetPlanReportMetadata, "accepted");
        if (datasetAccepted != reportAccepted) {
            throw new IllegalArgumentException(
                    "top-level datasetAccepted must match datasetPlanReport accepted");
        }

        String reportGateStatus = requiredString(datasetPlanReportMetadata, "gateStatus");
        if (!datasetGateStatus.equals(reportGateStatus)) {
            throw new IllegalArgumentException(
                    "top-level datasetGateStatus must match datasetPlanReport gateStatus");
        }
    }

    private static Map<String, Object> requiredMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return immutableMetadataMap(map, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static Map<String, Object> optionalMap(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return Map.of();
        }
        Object value = metadata.get(key);
        if (value instanceof Map<?, ?> map) {
            return immutableMetadataMap(map, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    private static Map<String, Object> immutableMetadataMap(Map<?, ?> values, String name) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            String key = mapKey(entry.getKey(), name);
            Object value = Objects.requireNonNull(entry.getValue(), name + " field '" + key + "' must not be null");
            copy.put(key, value);
        }
        return Collections.unmodifiableMap(copy);
    }

    private static String mapKey(Object key, String mapName) {
        if (key instanceof CharSequence text) {
            return requireText(text.toString(), mapName + " key");
        }
        throw new IllegalArgumentException(mapName + " keys must be strings");
    }

    private static String requiredString(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof CharSequence text) {
            return requireText(text.toString(), key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    private static long requiredLong(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            double numericValue = number.doubleValue();
            if (!Double.isFinite(numericValue)
                    || Math.rint(numericValue) != numericValue
                    || numericValue < Long.MIN_VALUE
                    || numericValue > Long.MAX_VALUE) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
            }
            return number.longValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Long.parseLong(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
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

    private static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
