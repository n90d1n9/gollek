package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Read-side view of a persisted checkpoint resume report sidecar.
 */
public record DiscreteTokenDatasetCheckpointResumeReportSnapshot(
        String status,
        boolean ready,
        boolean schemaAccepted,
        boolean datasetAccepted,
        boolean fingerprintMatched,
        boolean expectationAccepted,
        boolean currentPlanChecked,
        boolean currentPlanAccepted,
        String currentPlanGateStatus,
        DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
        boolean forceAccepted,
        List<String> compatibilityWarnings,
        boolean policyTracked,
        String runId,
        long checkpointStep,
        String schemaVersion,
        String datasetGateStatus,
        List<String> rejectionReasons,
        DiscreteTokenDatasetCheckpointResumeExpectation expectation,
        DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
        Map<String, Object> policyMetadata,
        Map<String, Object> currentPlanReportMetadata,
        DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint) {

    public DiscreteTokenDatasetCheckpointResumeReportSnapshot {
        status = requireText(status, "status");
        currentPlanGateStatus = requireText(currentPlanGateStatus, "currentPlanGateStatus");
        runId = requireText(runId, "runId");
        if (checkpointStep < 0L) {
            throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
        }
        schemaVersion = requireText(schemaVersion, "schemaVersion");
        datasetGateStatus = requireText(datasetGateStatus, "datasetGateStatus");
        rejectionReasons = immutableStringList(rejectionReasons, "rejectionReasons");
        compatibilityMode = Objects.requireNonNull(compatibilityMode, "compatibilityMode must not be null");
        compatibilityWarnings = immutableStringList(compatibilityWarnings, "compatibilityWarnings");
        expectation = Objects.requireNonNull(expectation, "expectation must not be null");
        fingerprintMatch = Objects.requireNonNull(fingerprintMatch, "fingerprintMatch must not be null");
        policyMetadata = immutableMetadataMap(policyMetadata, "policyMetadata");
        currentPlanReportMetadata = immutableMetadataMap(currentPlanReportMetadata, "currentPlanReportMetadata");
        checkpoint = Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        verifyConsistency(
                status,
                ready,
                schemaAccepted,
                datasetAccepted,
                fingerprintMatched,
                expectationAccepted,
                currentPlanChecked,
                currentPlanAccepted,
                currentPlanGateStatus,
                compatibilityMode,
                forceAccepted,
                compatibilityWarnings,
                policyTracked,
                runId,
                checkpointStep,
                schemaVersion,
                datasetGateStatus,
                rejectionReasons,
                expectation,
                fingerprintMatch,
                policyMetadata,
                currentPlanReportMetadata,
                checkpoint);
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot fromReport(
            DiscreteTokenDatasetCheckpointResumeReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return fromMetadata(report.toMetadata());
    }

    public static DiscreteTokenDatasetCheckpointResumeReportSnapshot fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeReportSnapshot(
                requiredString(metadata, "status"),
                requiredBoolean(metadata, "ready"),
                requiredBoolean(metadata, "schemaAccepted"),
                requiredBoolean(metadata, "datasetAccepted"),
                requiredBoolean(metadata, "fingerprintMatched"),
                requiredBoolean(metadata, "expectationAccepted"),
                requiredBoolean(metadata, "currentPlanChecked"),
                requiredBoolean(metadata, "currentPlanAccepted"),
                requiredString(metadata, "currentPlanGateStatus"),
                compatibilityModeFromMetadata(metadata),
                optionalBoolean(metadata, "forceAccepted", false),
                optionalStringList(metadata, "compatibilityWarnings"),
                requiredBoolean(metadata, "policyTracked"),
                requiredString(metadata, "runId"),
                requiredLong(metadata, "checkpointStep"),
                requiredString(metadata, "schemaVersion"),
                requiredString(metadata, "datasetGateStatus"),
                requiredStringList(metadata, "rejectionReasons"),
                DiscreteTokenDatasetCheckpointResumeExpectation.fromMetadata(requiredMap(metadata, "expectation")),
                DiscreteTokenDatasetFingerprintMatch.fromMetadata(requiredMap(metadata, "fingerprintMatch")),
                optionalMap(metadata, "policy"),
                optionalMap(metadata, "currentPlanReport"),
                DiscreteTokenDatasetCheckpointManifestSnapshot.fromMetadata(requiredMap(metadata, "checkpoint")));
    }

    public String summary() {
        String prefix = "checkpoint resume "
                + status
                + ": "
                + runId
                + " step "
                + checkpointStep
                + " dataset "
                + checkpoint.fingerprint().shortValue();
        if (ready) {
            return prefix;
        }
        return prefix + " (" + String.join("; ", rejectionReasons) + ")";
    }

    public void requireReady() {
        if (!ready) {
            throw new IllegalStateException(summary());
        }
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status);
        metadata.put("ready", ready);
        metadata.put("schemaAccepted", schemaAccepted);
        metadata.put("datasetAccepted", datasetAccepted);
        metadata.put("fingerprintMatched", fingerprintMatched);
        metadata.put("expectationAccepted", expectationAccepted);
        metadata.put("currentPlanChecked", currentPlanChecked);
        metadata.put("currentPlanAccepted", currentPlanAccepted);
        metadata.put("currentPlanGateStatus", currentPlanGateStatus);
        metadata.put("compatibilityMode", compatibilityMode.id());
        metadata.put("forceAccepted", forceAccepted);
        metadata.put("compatibilityWarnings", compatibilityWarnings);
        metadata.put("policyTracked", policyTracked);
        metadata.put("runId", runId);
        metadata.put("checkpointStep", checkpointStep);
        metadata.put("schemaVersion", schemaVersion);
        metadata.put("datasetGateStatus", datasetGateStatus);
        metadata.put("rejectionReasons", rejectionReasons);
        metadata.put("expectation", expectation.toMetadata());
        metadata.put("fingerprintMatch", fingerprintMatch.toMetadata());
        if (policyTracked) {
            metadata.put("policy", policyMetadata);
        }
        if (currentPlanChecked) {
            metadata.put("currentPlanReport", currentPlanReportMetadata);
        }
        metadata.put("checkpoint", checkpoint.toMetadata());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyConsistency(
            String status,
            boolean ready,
            boolean schemaAccepted,
            boolean datasetAccepted,
            boolean fingerprintMatched,
            boolean expectationAccepted,
            boolean currentPlanChecked,
            boolean currentPlanAccepted,
            String currentPlanGateStatus,
            DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityMode,
            boolean forceAccepted,
            List<String> compatibilityWarnings,
            boolean policyTracked,
            String runId,
            long checkpointStep,
            String schemaVersion,
            String datasetGateStatus,
            List<String> rejectionReasons,
            DiscreteTokenDatasetCheckpointResumeExpectation expectation,
            DiscreteTokenDatasetFingerprintMatch fingerprintMatch,
            Map<String, Object> policyMetadata,
            Map<String, Object> currentPlanReportMetadata,
            DiscreteTokenDatasetCheckpointManifestSnapshot checkpoint) {
        String expectedStatus = ready ? "ready" : "blocked";
        if (!expectedStatus.equals(status)) {
            throw new IllegalArgumentException("status must be " + expectedStatus + " when ready=" + ready);
        }
        if (ready && !rejectionReasons.isEmpty()) {
            throw new IllegalArgumentException("ready resume reports must not carry rejection reasons");
        }
        if (forceAccepted != (compatibilityMode.force() && ready && !compatibilityWarnings.isEmpty())) {
            throw new IllegalArgumentException("forceAccepted must match compatibility mode and warnings");
        }
        if (!runId.equals(checkpoint.runId())) {
            throw new IllegalArgumentException("runId must match checkpoint runId");
        }
        if (checkpointStep != checkpoint.checkpointStep()) {
            throw new IllegalArgumentException("checkpointStep must match checkpoint checkpointStep");
        }
        if (!schemaVersion.equals(checkpoint.schemaVersion())) {
            throw new IllegalArgumentException("schemaVersion must match checkpoint schemaVersion");
        }
        if (!datasetGateStatus.equals(checkpoint.datasetGateStatus())) {
            throw new IllegalArgumentException("datasetGateStatus must match checkpoint datasetGateStatus");
        }
        if (schemaAccepted != checkpoint.isCurrentSchema()) {
            throw new IllegalArgumentException("schemaAccepted must match checkpoint schema status");
        }
        if (datasetAccepted != checkpoint.datasetAccepted()) {
            throw new IllegalArgumentException("datasetAccepted must match checkpoint dataset status");
        }
        if (fingerprintMatched != fingerprintMatch.matches()) {
            throw new IllegalArgumentException("fingerprintMatched must match fingerprint comparison");
        }
        if (expectationAccepted != expectation.accepts(checkpoint)) {
            throw new IllegalArgumentException("expectationAccepted must match expectation result");
        }
        if (policyTracked != !policyMetadata.isEmpty()) {
            throw new IllegalArgumentException("policyTracked must match policy metadata presence");
        }
        if (currentPlanChecked != !currentPlanReportMetadata.isEmpty()) {
            throw new IllegalArgumentException("currentPlanChecked must match currentPlanReport presence");
        }
        verifyCurrentPlanReportConsistency(
                currentPlanChecked,
                currentPlanAccepted,
                currentPlanGateStatus,
                currentPlanReportMetadata);
        boolean computedReady = schemaAccepted
                && (datasetAccepted || compatibilityMode.allowRejectedCheckpointDataset())
                && (fingerprintMatched || compatibilityMode.allowFingerprintMismatch())
                && expectationAccepted
                && (currentPlanAccepted || compatibilityMode.allowCurrentPlanGateStatus(currentPlanGateStatus));
        if (ready != computedReady) {
            throw new IllegalArgumentException("ready must match resume report acceptance fields");
        }
    }

    private static void verifyCurrentPlanReportConsistency(
            boolean currentPlanChecked,
            boolean currentPlanAccepted,
            String currentPlanGateStatus,
            Map<String, Object> currentPlanReportMetadata) {
        if (!currentPlanChecked) {
            if (!"not-checked".equals(currentPlanGateStatus)) {
                throw new IllegalArgumentException(
                        "currentPlanGateStatus must be not-checked when currentPlanChecked is false");
            }
            if (!currentPlanAccepted) {
                throw new IllegalArgumentException(
                        "currentPlanAccepted must be true when currentPlanChecked is false");
            }
            return;
        }

        boolean reportAccepted = requiredBoolean(currentPlanReportMetadata, "accepted");
        if (currentPlanAccepted != reportAccepted) {
            throw new IllegalArgumentException("currentPlanAccepted must match currentPlanReport accepted");
        }
        String reportGateStatus = requiredString(currentPlanReportMetadata, "gateStatus");
        if (!currentPlanGateStatus.equals(reportGateStatus)) {
            throw new IllegalArgumentException("currentPlanGateStatus must match currentPlanReport gateStatus");
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

    private static List<String> requiredStringList(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof List<?> list) {
            return immutableStringList(list, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    private static List<String> optionalStringList(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return List.of();
        }
        Object value = metadata.get(key);
        if (value instanceof List<?> list) {
            return immutableStringList(list, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    private static List<String> immutableStringList(List<?> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof CharSequence text) {
                        return text.toString();
                    }
                    throw new IllegalArgumentException(name + " entries must be strings");
                })
                .toList();
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
        return booleanValue(value, key);
    }

    private static boolean optionalBoolean(Map<?, ?> metadata, String key, boolean defaultValue) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return defaultValue;
        }
        return booleanValue(metadata.get(key), key);
    }

    private static boolean booleanValue(Object value, String key) {
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

    private static DiscreteTokenDatasetCheckpointResumeCompatibilityMode compatibilityModeFromMetadata(
            Map<?, ?> metadata) {
        Object value = metadata.containsKey("compatibilityMode")
                ? metadata.get("compatibilityMode")
                : metadata.get("mode");
        return DiscreteTokenDatasetCheckpointResumeCompatibilityMode.fromMetadataValue(value);
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
