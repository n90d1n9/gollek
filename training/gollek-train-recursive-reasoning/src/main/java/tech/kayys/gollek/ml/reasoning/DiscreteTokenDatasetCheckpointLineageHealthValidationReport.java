package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Dependency-free validation result for checkpoint lineage health payloads.
 */
public record DiscreteTokenDatasetCheckpointLineageHealthValidationReport(
        boolean valid,
        String code,
        String message,
        String schemaId,
        String expectedKind,
        String expectedSchemaVersion,
        Map<String, Object> payloadSummary,
        List<String> errors) {

    public static final String VALID_CODE = "GOLLEK_LINEAGE_HEALTH_VALID";
    public static final String INVALID_CODE = "GOLLEK_LINEAGE_HEALTH_INVALID";
    public static final String INVALID_JSON_CODE = "GOLLEK_LINEAGE_HEALTH_JSON_INVALID";
    public static final String READ_ERROR_CODE = "GOLLEK_LINEAGE_HEALTH_READ_ERROR";
    public static final String KIND = "checkpoint-lineage-health-validation";
    public static final String SCHEMA_VERSION = "gollek.checkpoint-lineage-health-validation.v1";
    public static final String JSON_SCHEMA_ID =
            "https://gollek.ai/schemas/training/checkpoint-lineage-health-validation.v1.schema.json";
    public static final String JSON_SCHEMA_RESOURCE =
            "tech/kayys/gollek/ml/reasoning/schemas/checkpoint-lineage-health-validation.v1.schema.json";

    public DiscreteTokenDatasetCheckpointLineageHealthValidationReport {
        code = DiscreteTokenDatasetMetadataSupport.requireText(code, "code");
        message = DiscreteTokenDatasetMetadataSupport.requireText(message, "message");
        schemaId = DiscreteTokenDatasetMetadataSupport.requireText(schemaId, "schemaId");
        expectedKind = DiscreteTokenDatasetMetadataSupport.requireText(expectedKind, "expectedKind");
        expectedSchemaVersion =
                DiscreteTokenDatasetMetadataSupport.requireText(expectedSchemaVersion, "expectedSchemaVersion");
        payloadSummary = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(payloadSummary, "payloadSummary");
        errors = DiscreteTokenDatasetMetadataSupport.optionalTextList(errors, "errors");
        if (valid && !errors.isEmpty()) {
            throw new IllegalArgumentException("valid validation reports must not carry errors");
        }
        if (!valid && errors.isEmpty()) {
            throw new IllegalArgumentException("invalid validation reports must carry at least one error");
        }
        if (valid && !VALID_CODE.equals(code)) {
            throw new IllegalArgumentException("valid validation reports must use code " + VALID_CODE);
        }
        if (!valid && VALID_CODE.equals(code)) {
            throw new IllegalArgumentException("invalid validation reports must not use code " + VALID_CODE);
        }
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromSnapshot(
            DiscreteTokenDatasetCheckpointLineageHealthSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot must not be null");
        return new DiscreteTokenDatasetCheckpointLineageHealthValidationReport(
                true,
                VALID_CODE,
                "checkpoint lineage health payload is valid",
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                payloadSummary(snapshot),
                List.of());
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromMetadata(Map<?, ?> metadata) {
        try {
            if (metadata == null) {
                throw new NullPointerException("metadata must not be null");
            }
            return fromSnapshot(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.fromMetadata(metadata));
        } catch (RuntimeException e) {
            return invalid(
                    INVALID_CODE,
                    "checkpoint lineage health payload is invalid",
                    List.of(errorMessage(e)),
                    payloadSummary(metadata));
        }
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromJson(String json) {
        try {
            Map<String, Object> metadata = DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json);
            return fromMetadata(metadata);
        } catch (RuntimeException e) {
            return invalid(
                    INVALID_JSON_CODE,
                    "checkpoint lineage health JSON is invalid",
                    List.of(errorMessage(e)),
                    Map.of());
        }
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromPath(Path path) {
        try {
            Objects.requireNonNull(path, "path must not be null");
            return fromJson(Files.readString(path));
        } catch (IOException e) {
            return invalid(
                    READ_ERROR_CODE,
                    "checkpoint lineage health JSON could not be read",
                    List.of(e.getMessage()),
                    Map.of());
        } catch (UncheckedIOException e) {
            return invalid(
                    READ_ERROR_CODE,
                    "checkpoint lineage health JSON could not be read",
                    List.of(errorMessage(e)),
                    Map.of());
        } catch (RuntimeException e) {
            return invalid(
                    READ_ERROR_CODE,
                    "checkpoint lineage health JSON could not be read",
                    List.of(errorMessage(e)),
                    Map.of());
        }
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromValidationMetadata(
            Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        requireConstant(metadata, "kind", KIND);
        requireConstant(metadata, "schemaVersion", SCHEMA_VERSION);
        boolean valid = DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "valid");
        String status = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status");
        String expectedStatus = valid ? "valid" : "invalid";
        if (!expectedStatus.equals(status)) {
            throw new IllegalArgumentException("status must be '" + expectedStatus + "' when valid=" + valid);
        }
        return new DiscreteTokenDatasetCheckpointLineageHealthValidationReport(
                valid,
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "code"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "message"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "schemaId"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "expectedKind"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "expectedSchemaVersion"),
                DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(
                        DiscreteTokenDatasetMetadataSupport.requiredMap(metadata, "payloadSummary"),
                        "payloadSummary"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "errors"));
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromValidationJson(String json) {
        return fromValidationMetadata(DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json));
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport fromValidationPath(Path path) {
        try {
            Objects.requireNonNull(path, "path must not be null");
            return fromValidationJson(Files.readString(path));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read checkpoint lineage health validation report", e);
        }
    }

    public static String jsonSchemaText() {
        return DiscreteTokenDatasetSchemaContract.resourceText(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.class,
                JSON_SCHEMA_RESOURCE,
                "checkpoint lineage health validation JSON schema");
    }

    public static Map<String, Object> jsonSchemaMetadata() {
        return DiscreteTokenDatasetSchemaContract.jsonMetadata(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.class,
                JSON_SCHEMA_RESOURCE,
                "checkpoint lineage health validation JSON schema");
    }

    public String status() {
        return valid ? "valid" : "invalid";
    }

    public boolean invalid() {
        return !valid;
    }

    public void requireValid() {
        if (!valid) {
            throw new IllegalStateException(message + ": " + String.join("; ", errors));
        }
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", KIND);
        metadata.put("schemaVersion", SCHEMA_VERSION);
        metadata.put("status", status());
        metadata.put("valid", valid);
        metadata.put("code", code);
        metadata.put("message", message);
        metadata.put("schemaId", schemaId);
        metadata.put("expectedKind", expectedKind);
        metadata.put("expectedSchemaVersion", expectedSchemaVersion);
        metadata.put("payloadSummary", payloadSummary);
        metadata.put("errors", errors);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static DiscreteTokenDatasetCheckpointLineageHealthValidationReport invalid(
            String code,
            String message,
            List<String> errors,
            Map<String, Object> payloadSummary) {
        return new DiscreteTokenDatasetCheckpointLineageHealthValidationReport(
                false,
                code,
                message,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                payloadSummary,
                errors);
    }

    private static Map<String, Object> payloadSummary(DiscreteTokenDatasetCheckpointLineageHealthSnapshot snapshot) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("kind", snapshot.kind());
        summary.put("schemaVersion", snapshot.schemaVersion());
        summary.put("status", snapshot.status());
        summary.put("healthy", snapshot.healthy());
        summary.put("healthScore", snapshot.healthScore());
        summary.put("issueCount", snapshot.issueCount());
        summary.put("issueDetailCount", snapshot.issueDetailCount());
        summary.put("blockingIssueCount", snapshot.blockingIssueCount());
        summary.put("failingCheckCount", snapshot.failingCheckCount());
        snapshot.primaryIssueCode().ifPresent(code -> summary.put("primaryIssueCode", code));
        snapshot.primaryIssueType().ifPresent(type -> summary.put("primaryIssueType", type));
        snapshot.primaryFailingCheckCode().ifPresent(code -> summary.put("primaryFailingCheckCode", code));
        snapshot.primaryFailingCheckType().ifPresent(type -> summary.put("primaryFailingCheckType", type));
        summary.put("summary", snapshot.summary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(summary));
    }

    private static Map<String, Object> payloadSummary(Map<?, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        copyIfPresent(metadata, summary, "kind");
        copyIfPresent(metadata, summary, "schemaVersion");
        copyIfPresent(metadata, summary, "status");
        copyIfPresent(metadata, summary, "healthy");
        copyIfPresent(metadata, summary, "healthScore");
        copyIfPresent(metadata, summary, "issueCount");
        copyIfPresent(metadata, summary, "issueDetailCount");
        copyIfPresent(metadata, summary, "blockingIssueCount");
        copyIfPresent(metadata, summary, "failingCheckCount");
        copyIfPresent(metadata, summary, "primaryIssueCode");
        copyIfPresent(metadata, summary, "primaryIssueType");
        copyIfPresent(metadata, summary, "primaryFailingCheckCode");
        copyIfPresent(metadata, summary, "primaryFailingCheckType");
        copyIfPresent(metadata, summary, "summary");
        return Collections.unmodifiableMap(new LinkedHashMap<>(summary));
    }

    private static void copyIfPresent(Map<?, ?> source, Map<String, Object> target, String key) {
        if (source.containsKey(key) && source.get(key) != null) {
            target.put(key, source.get(key));
        }
    }

    private static String errorMessage(Throwable error) {
        String message = error.getMessage();
        if (message == null || message.isBlank()) {
            return error.getClass().getSimpleName();
        }
        return message;
    }

    private static void requireConstant(Map<?, ?> metadata, String key, String expected) {
        String actual = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, key);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(key + " must be " + expected + " but was " + actual);
        }
    }

}
