package tech.kayys.gollek.ml.reasoning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Schema catalog for checkpoint lineage health payload contracts.
 */
public final class DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog {
    public static final String KIND = "checkpoint-lineage-health-schema-catalog";
    public static final String SCHEMA_VERSION = "gollek.checkpoint-lineage-health-schema-catalog.v1";
    public static final String LOCK_KIND = "checkpoint-lineage-health-schema-lock";
    public static final String LOCK_SCHEMA_VERSION = "gollek.checkpoint-lineage-health-schema-lock.v1";
    public static final String LOCK_VALIDATION_KIND = "checkpoint-lineage-health-schema-lock-validation";
    public static final String LOCK_VALIDATION_SCHEMA_VERSION =
            "gollek.checkpoint-lineage-health-schema-lock-validation.v1";
    public static final String JSON_SCHEMA_ID =
            "https://gollek.ai/schemas/training/checkpoint-lineage-health-schema-catalog.v1.schema.json";
    public static final String JSON_SCHEMA_RESOURCE =
            "tech/kayys/gollek/ml/reasoning/schemas/checkpoint-lineage-health-schema-catalog.v1.schema.json";
    public static final String LOCK_JSON_SCHEMA_ID =
            "https://gollek.ai/schemas/training/checkpoint-lineage-health-schema-lock.v1.schema.json";
    public static final String LOCK_JSON_SCHEMA_RESOURCE =
            "tech/kayys/gollek/ml/reasoning/schemas/checkpoint-lineage-health-schema-lock.v1.schema.json";
    public static final String LOCK_VALIDATION_JSON_SCHEMA_ID =
            "https://gollek.ai/schemas/training/checkpoint-lineage-health-schema-lock-validation.v1.schema.json";
    public static final String LOCK_VALIDATION_JSON_SCHEMA_RESOURCE =
            "tech/kayys/gollek/ml/reasoning/schemas/checkpoint-lineage-health-schema-lock-validation.v1.schema.json";

    private static final DiscreteTokenDatasetSchemaContract HEALTH_CONTRACT =
            new DiscreteTokenDatasetSchemaContract(
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.class,
                    "checkpointLineageHealth",
                    "Checkpoint lineage health",
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                    DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_RESOURCE,
                    List.of("lineageHealth", "lineageHealthSchema"));
    private static final DiscreteTokenDatasetSchemaContract HEALTH_VALIDATION_CONTRACT =
            new DiscreteTokenDatasetSchemaContract(
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.class,
                    "checkpointLineageHealthValidation",
                    "Checkpoint lineage health validation report",
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.KIND,
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.SCHEMA_VERSION,
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_ID,
                    DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_RESOURCE,
                    List.of("lineageHealthValidation", "lineageHealthValidationSchema"));
    private static final DiscreteTokenDatasetSchemaBundle BUNDLE =
            DiscreteTokenDatasetSchemaBundle
                    .builder(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.class)
                    .catalog(KIND, SCHEMA_VERSION, JSON_SCHEMA_ID, JSON_SCHEMA_RESOURCE)
                    .lock(LOCK_KIND, LOCK_SCHEMA_VERSION, LOCK_JSON_SCHEMA_ID, LOCK_JSON_SCHEMA_RESOURCE)
                    .lockValidation(
                            LOCK_VALIDATION_KIND,
                            LOCK_VALIDATION_SCHEMA_VERSION,
                            LOCK_VALIDATION_JSON_SCHEMA_ID,
                            LOCK_VALIDATION_JSON_SCHEMA_RESOURCE)
                    .labels(
                            "checkpoint lineage health schema catalog",
                            "checkpoint lineage health schema lock",
                            "checkpoint lineage health schema lock validation")
                    .codes(
                            "GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_VALID",
                            "GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_MISMATCH")
                    .addContract(HEALTH_CONTRACT)
                    .addContract(HEALTH_VALIDATION_CONTRACT)
                    .build();

    private DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog() {}

    public static Map<String, Object> catalogMetadata() {
        return catalogMetadata(false);
    }

    public static Map<String, Object> catalogMetadata(boolean includeSchemas) {
        return BUNDLE.catalogMetadata(includeSchemas);
    }

    public static DiscreteTokenDatasetSchemaCatalogSnapshot catalogSnapshot() {
        return BUNDLE.catalogSnapshot();
    }

    public static DiscreteTokenDatasetSchemaCatalogSnapshot catalogSnapshot(boolean includeSchemas) {
        return BUNDLE.catalogSnapshot(includeSchemas);
    }

    public static Map<String, Object> lockMetadata() {
        return BUNDLE.lockMetadata();
    }

    public static DiscreteTokenDatasetSchemaLockSnapshot lockSnapshot() {
        return BUNDLE.lockSnapshot();
    }

    public static Map<String, Object> validateCurrentLock() {
        return BUNDLE.validateCurrentLock();
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport validateCurrentLockReport() {
        return BUNDLE.validateCurrentLockReport();
    }

    public static Map<String, Object> requireCurrentLockValid() {
        return BUNDLE.requireCurrentLockValid();
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport requireCurrentLockValidReport() {
        return BUNDLE.requireCurrentLockValidReport();
    }

    public static Map<String, Object> requireValidLockMetadata(Map<?, ?> expectedLock) {
        return BUNDLE.requireValidLockMetadata(expectedLock);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport validateLockMetadataReport(
            Map<?, ?> expectedLock) {
        return BUNDLE.validateLockMetadataReport(expectedLock);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport requireValidLockMetadataReport(
            Map<?, ?> expectedLock) {
        return BUNDLE.requireValidLockMetadataReport(expectedLock);
    }

    public static Map<String, Object> validateLockSnapshot(DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return BUNDLE.validateLockSnapshot(expectedLock);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport validateLockSnapshotReport(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return BUNDLE.validateLockSnapshotReport(expectedLock);
    }

    public static Map<String, Object> requireValidLockSnapshot(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return BUNDLE.requireValidLockSnapshot(expectedLock);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport requireValidLockSnapshotReport(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return BUNDLE.requireValidLockSnapshotReport(expectedLock);
    }

    public static Map<String, Object> validateLockJson(String expectedLockJson) {
        return BUNDLE.validateLockJson(expectedLockJson);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport validateLockJsonReport(String expectedLockJson) {
        return BUNDLE.validateLockJsonReport(expectedLockJson);
    }

    public static Map<String, Object> requireValidLockJson(String expectedLockJson) {
        return BUNDLE.requireValidLockJson(expectedLockJson);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport requireValidLockJsonReport(
            String expectedLockJson) {
        return BUNDLE.requireValidLockJsonReport(expectedLockJson);
    }

    public static Map<String, Object> validateLockPath(Path expectedLockPath) throws IOException {
        return BUNDLE.validateLockPath(expectedLockPath);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport validateLockPathReport(Path expectedLockPath)
            throws IOException {
        return BUNDLE.validateLockPathReport(expectedLockPath);
    }

    public static Map<String, Object> requireValidLockPath(Path expectedLockPath) throws IOException {
        return BUNDLE.requireValidLockPath(expectedLockPath);
    }

    public static DiscreteTokenDatasetSchemaLockValidationReport requireValidLockPathReport(Path expectedLockPath)
            throws IOException {
        return BUNDLE.requireValidLockPathReport(expectedLockPath);
    }

    public static void requireValidLockValidation(Map<?, ?> validation) {
        BUNDLE.requireValidLockValidation(validation);
    }

    public static Map<String, Object> validateLockMetadata(Map<?, ?> expectedLock) {
        return BUNDLE.validateLockMetadata(expectedLock);
    }

    public static List<Map<String, Object>> contractsMetadata() {
        return contractsMetadata(false);
    }

    public static List<Map<String, Object>> contractsMetadata(boolean includeSchemas) {
        return BUNDLE.contractsMetadata(includeSchemas);
    }

    public static String jsonSchemaText() {
        return BUNDLE.jsonSchemaText();
    }

    public static Map<String, Object> jsonSchemaMetadata() {
        return BUNDLE.jsonSchemaMetadata();
    }

    public static String lockJsonSchemaText() {
        return BUNDLE.lockJsonSchemaText();
    }

    public static Map<String, Object> lockJsonSchemaMetadata() {
        return BUNDLE.lockJsonSchemaMetadata();
    }

    public static String lockValidationJsonSchemaText() {
        return BUNDLE.lockValidationJsonSchemaText();
    }

    public static Map<String, Object> lockValidationJsonSchemaMetadata() {
        return BUNDLE.lockValidationJsonSchemaMetadata();
    }
}
