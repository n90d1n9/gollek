package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DiscreteTokenDatasetCheckpointLineageHealthSnapshotTest {
    private static final String FIXTURE_ROOT = "tech/kayys/gollek/ml/reasoning/fixtures/";

    @Test
    void exposesJsonSchemaResource() {
        Map<String, Object> schema = DiscreteTokenDatasetCheckpointLineageHealthSnapshot.jsonSchemaMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, schema.get("$id"));
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.get("$schema"));
        assertEquals("object", schema.get("type"));
        assertTrue(((List<?>) schema.get("required")).contains("kind"));
        assertTrue(((List<?>) schema.get("required")).contains("schemaVersion"));
        assertTrue(((List<?>) schema.get("required")).contains("checkSummary"));
        assertTrue(((Map<?, ?>) schema.get("properties")).containsKey("primaryFailingCheck"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("check"));
    }

    @Test
    void exposesValidationReportJsonSchemaResource() {
        Map<String, Object> schema = DiscreteTokenDatasetCheckpointLineageHealthValidationReport.jsonSchemaMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_ID, schema.get("$id"));
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.get("$schema"));
        assertEquals("object", schema.get("type"));
        assertTrue(((List<?>) schema.get("required")).contains("valid"));
        assertTrue(((List<?>) schema.get("required")).contains("payloadSummary"));
        assertTrue(((Map<?, ?>) schema.get("properties")).containsKey("expectedSchemaVersion"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("payloadSummary"));
    }

    @Test
    void exposesLineageHealthSchemaCatalogJsonSchemaResource() {
        Map<String, Object> schema = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.jsonSchemaMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.JSON_SCHEMA_ID, schema.get("$id"));
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.get("$schema"));
        assertEquals("object", schema.get("type"));
        assertTrue(((List<?>) schema.get("required")).contains("contracts"));
        assertTrue(((List<?>) schema.get("required")).contains("catalogSchemaSha256"));
        assertTrue(((List<?>) schema.get("required")).contains("schemaSetSha256"));
        assertTrue(((Map<?, ?>) schema.get("properties")).containsKey("schemaCount"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("contract"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("sha256"));
    }

    @Test
    void schemaContractUtilityExposesStableMetadata() {
        DiscreteTokenDatasetSchemaContract contract = new DiscreteTokenDatasetSchemaContract(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.class,
                "checkpointLineageHealth",
                "Checkpoint lineage health",
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_RESOURCE,
                List.of("lineageHealth", "lineageHealthSchema"));
        Map<String, Object> metadata = contract.toMetadata(true);

        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.jsonSchemaText(),
                contract.jsonSchemaText());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                contract.jsonSchemaMetadata().get("$id"));
        assertEquals(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                DiscreteTokenDatasetSchemaContract.sha256Hex("abc"));
        assertEquals(3, DiscreteTokenDatasetSchemaContract.utf8ByteCount("abc"));
        assertEquals("checkpointLineageHealth", metadata.get("name"));
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND, metadata.get("payloadKind"));
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, metadata.get("jsonSchemaId"));
        assertEquals(64, String.valueOf(metadata.get("jsonSchemaSha256")).length());
        assertTrue(((Number) metadata.get("jsonSchemaByteCount")).intValue() > 0);
        assertTrue(((List<?>) metadata.get("inspectorSections")).contains("lineageHealthSchema"));
        assertTrue(metadata.containsKey("schema"));
    }

    @Test
    void schemaBundleUtilityPreservesLineageHealthCatalogOutputs() {
        DiscreteTokenDatasetSchemaBundle bundle = lineageHealthSchemaBundle();
        Map<String, Object> mismatchedLock = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());
        mismatchedLock.put("schemaSetSha256", "0000000000000000000000000000000000000000000000000000000000000000");

        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(),
                bundle.catalogMetadata());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(true),
                bundle.catalogMetadata(true));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata(),
                bundle.lockMetadata());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock(),
                bundle.validateCurrentLock());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(mismatchedLock),
                bundle.validateLockMetadata(mismatchedLock));
        assertThrows(IllegalStateException.class, () -> bundle.requireValidLockMetadata(mismatchedLock));
    }

    @Test
    void schemaBundleBuilderRejectsDuplicateContractNames() {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> lineageHealthSchemaBundleBuilder()
                        .addContract(lineageHealthContract())
                        .addContract(lineageHealthContract())
                        .build());

        assertTrue(error.getMessage().contains("duplicate contract name"));
        assertTrue(error.getMessage().contains("checkpointLineageHealth"));
    }

    @Test
    void schemaBundleExposesCatalogSnapshot(@TempDir Path tempDir) throws IOException {
        DiscreteTokenDatasetSchemaBundle bundle = lineageHealthSchemaBundle();
        Map<String, Object> compactCatalog =
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata();
        Map<String, Object> embeddedCatalog =
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(true);
        String embeddedJson = DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(embeddedCatalog);
        Path embeddedPath = tempDir.resolve("lineage-health-schema-catalog.json");
        Files.writeString(embeddedPath, embeddedJson, StandardCharsets.UTF_8);
        DiscreteTokenDatasetSchemaCatalogSnapshot compactSnapshot = bundle.catalogSnapshot();
        DiscreteTokenDatasetSchemaCatalogSnapshot embeddedSnapshot = bundle.catalogSnapshot(true);
        DiscreteTokenDatasetSchemaCatalogSnapshot jsonSnapshot =
                DiscreteTokenDatasetSchemaCatalogSnapshot.fromJson(embeddedJson);
        DiscreteTokenDatasetSchemaCatalogSnapshot pathSnapshot =
                DiscreteTokenDatasetSchemaCatalogSnapshot.fromPath(embeddedPath);

        assertMetadataEquals(compactCatalog, compactSnapshot.toMetadata());
        assertMetadataEquals(embeddedCatalog, embeddedSnapshot.toMetadata());
        assertMetadataEquals(embeddedCatalog, jsonSnapshot.toMetadata());
        assertMetadataEquals(embeddedCatalog, pathSnapshot.toMetadata());
        assertMetadataEquals(
                compactCatalog,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogSnapshot().toMetadata());
        assertEquals("checkpoint-lineage-health-schema-catalog", compactSnapshot.kind());
        assertEquals("gollek.checkpoint-lineage-health-schema-catalog.v1", compactSnapshot.schemaVersion());
        assertEquals(2, compactSnapshot.schemaCount());
        assertEquals(64, compactSnapshot.schemaSetIdentity().schemaSetSha256().length());
        assertEquals(2, compactSnapshot.schemaSetIdentity().schemaCount());
        assertEquals(
                Map.of(
                        "schemaSetSha256",
                        compactSnapshot.schemaSetSha256(),
                        "schemaCount",
                        compactSnapshot.schemaCount()),
                compactSnapshot.schemaSetIdentity().toMetadata());
        assertEquals(List.of("checkpointLineageHealth", "checkpointLineageHealthValidation"),
                compactSnapshot.contractNames());
        assertTrue(compactSnapshot.contract("checkpointLineageHealth").isPresent());
        assertEquals(2, compactSnapshot.contractDetails().size());
        assertFalse(compactSnapshot.contract("missing").isPresent());
        assertTrue(compactSnapshot.contractDetail("missing").isEmpty());
        assertFalse(compactSnapshot.hasEmbeddedSchemas());
        assertTrue(embeddedSnapshot.hasEmbeddedSchemas());
        assertTrue(embeddedSnapshot.contract("checkpointLineageHealth").orElseThrow().containsKey("schema"));
        DiscreteTokenDatasetSchemaContractSnapshot compactContract =
                compactSnapshot.contractDetail("checkpointLineageHealth").orElseThrow();
        DiscreteTokenDatasetSchemaContractSnapshot embeddedContract =
                embeddedSnapshot.contractDetail("checkpointLineageHealth").orElseThrow();
        assertEquals(compactSnapshot.contract("checkpointLineageHealth").orElseThrow(), compactContract.toMetadata());
        assertEquals(compactContract, compactSnapshot.contractsByName().get("checkpointLineageHealth"));
        assertEquals(compactContract, compactSnapshot.requireContractDetail("checkpointLineageHealth"));
        assertEquals(compactContract.toMetadata(), compactSnapshot.requireContract("checkpointLineageHealth"));
        assertEquals(
                compactContract,
                compactSnapshot.contractsByJsonSchemaId().get(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertEquals(
                compactContract,
                compactSnapshot.contractForJsonSchemaId(
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID)
                        .orElseThrow());
        assertEquals(
                compactContract,
                compactSnapshot.requireContractForJsonSchemaId(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertTrue(compactContract.isJsonSchema(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertTrue(compactSnapshot.contractForJsonSchemaId("missing").isEmpty());
        assertEquals(
                List.of(compactContract),
                compactSnapshot.contractsByPayloadKind().get(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND));
        assertEquals(
                List.of(compactContract),
                compactSnapshot.contractsForPayloadKind(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND));
        assertEquals(List.of(), compactSnapshot.contractsForPayloadKind("missing"));
        assertEquals(
                compactContract,
                compactSnapshot.contractForPayloadSchema(
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION)
                        .orElseThrow());
        assertEquals(
                compactContract,
                compactSnapshot.requireContractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION));
        assertTrue(compactSnapshot.contractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        "missing")
                .isEmpty());
        assertEquals("checkpointLineageHealth", compactContract.name());
        assertEquals("Checkpoint lineage health", compactContract.label().orElseThrow());
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND, compactContract.payloadKind());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                compactContract.payloadSchemaVersion());
        assertTrue(compactContract.isPayloadSchema(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION));
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, compactContract.jsonSchemaId());
        assertTrue(compactContract.jsonSchemaResource().orElseThrow().endsWith("checkpoint-lineage-health.v1.schema.json"));
        assertTrue(compactContract.jsonSchemaByteCount() > 0);
        assertEquals("https://json-schema.org/draft/2020-12/schema", compactContract.jsonSchemaDraft().orElseThrow());
        assertTrue(compactContract.title().orElseThrow().contains("checkpoint lineage health"));
        assertTrue(compactContract.description().orElseThrow().contains("recursive"));
        assertTrue(compactContract.inspectorSections().contains("lineageHealthSchema"));
        assertFalse(compactContract.hasEmbeddedSchema());
        assertTrue(compactContract.schemaMetadata().isEmpty());
        assertTrue(embeddedContract.hasEmbeddedSchema());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                embeddedContract.schemaMetadata().orElseThrow().get("$id"));
        IllegalArgumentException missingCatalogContract = assertThrows(
                IllegalArgumentException.class,
                () -> compactSnapshot.requireContractDetail("missing"));
        assertTrue(missingCatalogContract.getMessage().contains("schema catalog contract not found: missing"));
        IllegalArgumentException missingCatalogJsonSchema = assertThrows(
                IllegalArgumentException.class,
                () -> compactSnapshot.requireContractForJsonSchemaId("missing"));
        assertTrue(missingCatalogJsonSchema.getMessage()
                .contains("schema catalog contract not found for JSON schema id: missing"));
        IllegalArgumentException missingCatalogPayloadSchema = assertThrows(
                IllegalArgumentException.class,
                () -> compactSnapshot.requireContractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        "missing"));
        assertTrue(missingCatalogPayloadSchema.getMessage()
                .contains("schema catalog contract not found for payload schema"));
        compactSnapshot.requireCurrentSchema(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.SCHEMA_VERSION,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.JSON_SCHEMA_ID);
        assertThrows(
                IllegalStateException.class,
                () -> compactSnapshot.requireCurrentSchema(
                        "wrong",
                        compactSnapshot.schemaVersion(),
                        compactSnapshot.catalogSchemaId()));
    }

    @Test
    void schemaBundleValidatesLockJsonAndPath(@TempDir Path tempDir) throws IOException {
        DiscreteTokenDatasetSchemaBundle bundle = lineageHealthSchemaBundle();
        Map<String, Object> lock = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata();
        Map<String, Object> mismatchedLock = new LinkedHashMap<>(lock);
        mismatchedLock.put("schemaSetSha256", "0000000000000000000000000000000000000000000000000000000000000000");
        String lockJson = DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(lock);
        Path lockPath = tempDir.resolve("lineage-health-schema-lock.json");
        Files.writeString(lockPath, lockJson, StandardCharsets.UTF_8);
        DiscreteTokenDatasetSchemaLockSnapshot lockSnapshot = bundle.lockSnapshot();
        DiscreteTokenDatasetSchemaLockSnapshot jsonLockSnapshot =
                DiscreteTokenDatasetSchemaLockSnapshot.fromJson(lockJson);
        DiscreteTokenDatasetSchemaLockSnapshot pathLockSnapshot =
                DiscreteTokenDatasetSchemaLockSnapshot.fromPath(lockPath);
        DiscreteTokenDatasetSchemaLockValidationReport pathReport = bundle.validateLockPathReport(lockPath);
        DiscreteTokenDatasetSchemaLockValidationReport mismatchReport =
                bundle.validateLockMetadataReport(mismatchedLock);

        assertMetadataEquals(lock, lockSnapshot.toMetadata());
        assertMetadataEquals(lock, jsonLockSnapshot.toMetadata());
        assertMetadataEquals(lock, pathLockSnapshot.toMetadata());
        assertMetadataEquals(lock, DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockSnapshot().toMetadata());
        assertEquals("checkpoint-lineage-health-schema-lock", lockSnapshot.kind());
        assertEquals("gollek.checkpoint-lineage-health-schema-lock.v1", lockSnapshot.schemaVersion());
        assertEquals(2, lockSnapshot.schemaCount());
        assertEquals(64, lockSnapshot.schemaSetIdentity().schemaSetSha256().length());
        assertEquals(2, lockSnapshot.schemaSetIdentity().schemaCount());
        assertTrue(lockSnapshot.matchesSchemaSet(
                DiscreteTokenDatasetSchemaSetIdentity.fromCatalogSnapshot(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogSnapshot())));
        assertTrue(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog
                .catalogSnapshot()
                .matchesSchemaSet(lockSnapshot));
        assertEquals(
                lockSnapshot.schemaSetIdentity(),
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogSnapshot().requireSchemaSet(lockSnapshot));
        assertEquals(
                lockSnapshot.schemaSetIdentity(),
                lockSnapshot.requireSchemaSet(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogSnapshot()));
        assertEquals(List.of("checkpointLineageHealth", "checkpointLineageHealthValidation"),
                lockSnapshot.contractNames());
        assertTrue(lockSnapshot.contract("checkpointLineageHealth").isPresent());
        assertEquals(2, lockSnapshot.contractDetails().size());
        assertFalse(lockSnapshot.contract("missing").isPresent());
        assertTrue(lockSnapshot.contractDetail("missing").isEmpty());
        DiscreteTokenDatasetSchemaContractSnapshot lockContract =
                lockSnapshot.contractDetail("checkpointLineageHealth").orElseThrow();
        assertEquals(lockSnapshot.contract("checkpointLineageHealth").orElseThrow(), lockContract.toMetadata());
        assertEquals(lockContract, lockSnapshot.contractsByName().get("checkpointLineageHealth"));
        assertEquals(lockContract, lockSnapshot.requireContractDetail("checkpointLineageHealth"));
        assertEquals(lockContract.toMetadata(), lockSnapshot.requireContract("checkpointLineageHealth"));
        assertEquals(
                lockContract,
                lockSnapshot.contractsByJsonSchemaId().get(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertEquals(
                lockContract,
                lockSnapshot.contractForJsonSchemaId(
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID)
                        .orElseThrow());
        assertEquals(
                lockContract,
                lockSnapshot.requireContractForJsonSchemaId(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertTrue(lockContract.isJsonSchema(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID));
        assertTrue(lockSnapshot.contractForJsonSchemaId("missing").isEmpty());
        assertEquals(
                List.of(lockContract),
                lockSnapshot.contractsByPayloadKind().get(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND));
        assertEquals(
                List.of(lockContract),
                lockSnapshot.contractsForPayloadKind(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND));
        assertEquals(List.of(), lockSnapshot.contractsForPayloadKind("missing"));
        assertEquals(
                lockContract,
                lockSnapshot.contractForPayloadSchema(
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION)
                        .orElseThrow());
        assertEquals(
                lockContract,
                lockSnapshot.requireContractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION));
        assertTrue(lockSnapshot.contractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        "missing")
                .isEmpty());
        assertEquals("checkpointLineageHealth", lockContract.name());
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND, lockContract.payloadKind());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                lockContract.payloadSchemaVersion());
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, lockContract.jsonSchemaId());
        assertTrue(lockContract.label().isEmpty());
        assertTrue(lockContract.jsonSchemaResource().isEmpty());
        assertTrue(lockContract.inspectorSections().isEmpty());
        assertFalse(lockContract.hasEmbeddedSchema());
        IllegalArgumentException missingLockContract = assertThrows(
                IllegalArgumentException.class,
                () -> lockSnapshot.requireContractDetail("missing"));
        assertTrue(missingLockContract.getMessage().contains("schema lock contract not found: missing"));
        IllegalArgumentException missingLockJsonSchema = assertThrows(
                IllegalArgumentException.class,
                () -> lockSnapshot.requireContractForJsonSchemaId("missing"));
        assertTrue(missingLockJsonSchema.getMessage()
                .contains("schema lock contract not found for JSON schema id: missing"));
        IllegalArgumentException missingLockPayloadSchema = assertThrows(
                IllegalArgumentException.class,
                () -> lockSnapshot.requireContractForPayloadSchema(
                        DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                        "missing"));
        assertTrue(missingLockPayloadSchema.getMessage()
                .contains("schema lock contract not found for payload schema"));
        lockSnapshot.requireCurrentSchema(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_SCHEMA_VERSION,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_JSON_SCHEMA_ID);
        assertThrows(
                IllegalStateException.class,
                () -> lockSnapshot.requireCurrentSchema(
                        "wrong",
                        lockSnapshot.schemaVersion(),
                        lockSnapshot.lockSchemaId()));
        assertMetadataEquals(bundle.validateLockMetadata(lock), bundle.validateLockJson(lockJson));
        assertMetadataEquals(bundle.validateLockMetadata(lock), bundle.validateLockPath(lockPath));
        assertMetadataEquals(bundle.validateLockMetadata(lock), bundle.validateLockSnapshot(lockSnapshot));
        assertMetadataEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(lock),
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockJson(lockJson));
        assertMetadataEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(lock),
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockPath(lockPath));
        assertMetadataEquals(bundle.validateLockMetadata(lock), bundle.requireValidLockJson(lockJson));
        assertMetadataEquals(bundle.validateLockMetadata(lock), bundle.requireValidLockPath(lockPath));
        assertMetadataEquals(bundle.validateLockMetadata(lock), pathReport.toMetadata());
        assertMetadataEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock(),
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLockReport().toMetadata());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock(),
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog
                        .validateLockSnapshotReport(lockSnapshot)
                        .toMetadata());
        assertTrue(pathReport.valid());
        assertEquals("valid", pathReport.status());
        assertEquals("GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_VALID", pathReport.code());
        assertEquals(0, pathReport.mismatchCount());
        assertEquals(pathReport.expectedSchemaSetIdentity(), pathReport.actualSchemaSetIdentity());
        assertEquals(lockSnapshot.schemaSetIdentity(), pathReport.actualSchemaSetIdentity());
        assertTrue(pathReport.schemaSetIdentityMatch());
        assertEquals(pathReport.expectedSchemaSetIdentity(), pathReport.requireSchemaSetIdentityMatch());
        assertFalse(pathReport.hasMismatches());
        assertTrue(pathReport.mismatchDetails().isEmpty());
        assertTrue(pathReport.firstMismatch().isEmpty());
        assertTrue(pathReport.mismatchTypes().isEmpty());
        assertTrue(pathReport.mismatchesByPath().isEmpty());
        assertTrue(pathReport.mismatchesByType().isEmpty());
        pathReport.requireValid();
        assertFalse(mismatchReport.valid());
        assertEquals("invalid", mismatchReport.status());
        assertEquals("GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_MISMATCH", mismatchReport.code());
        assertEquals(1, mismatchReport.mismatchCount());
        assertFalse(mismatchReport.schemaSetIdentityMatch());
        assertThrows(IllegalStateException.class, mismatchReport::requireSchemaSetIdentityMatch);
        assertTrue(mismatchReport.hasMismatches());
        assertEquals(List.of("schemaSetSha256"), mismatchReport.mismatchPaths());
        assertEquals(List.of("schemaSetSha256"), mismatchReport.mismatchTypes());
        assertTrue(mismatchReport.hasMismatchAtPath("schemaSetSha256"));
        assertFalse(mismatchReport.hasMismatchAtPath("missing"));
        assertTrue(mismatchReport.hasMismatchType("schemaSetSha256"));
        assertFalse(mismatchReport.hasMismatchType("missing"));
        DiscreteTokenDatasetSchemaLockMismatch mismatch = mismatchReport.firstMismatch().orElseThrow();
        assertEquals(mismatchReport.mismatches().get(0), mismatch.toMetadata());
        assertEquals("schemaSetSha256", mismatch.type());
        assertEquals("schemaSetSha256", mismatch.path());
        assertEquals(mismatchedLock.get("schemaSetSha256"), mismatch.expected());
        assertEquals(lock.get("schemaSetSha256"), mismatch.actual());
        assertEquals("schemaSetSha256 (schemaSetSha256)", mismatch.summary());
        assertEquals(List.of(mismatch), mismatchReport.mismatchesForPath("schemaSetSha256"));
        assertEquals(List.of(), mismatchReport.mismatchesForPath("missing"));
        assertEquals(List.of(mismatch), mismatchReport.mismatchesForType("schemaSetSha256"));
        assertEquals(List.of(), mismatchReport.mismatchesForType("missing"));
        assertEquals(List.of(mismatch), mismatchReport.mismatchesByPath().get("schemaSetSha256"));
        assertEquals(List.of(mismatch), mismatchReport.mismatchesByType().get("schemaSetSha256"));
        assertThrows(IllegalStateException.class, mismatchReport::requireValid);
    }

    @Test
    void schemaSnapshotReadersRejectMalformedMetadata() {
        Map<String, Object> catalog = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata());
        Map<String, Object> lock = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());
        Map<String, Object> report = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock());
        Map<String, Object> mismatchedLock = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());
        Map<String, Object> duplicateCatalog = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata());
        Map<String, Object> duplicateLock = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());
        Map<String, Object> duplicateJsonSchemaCatalog = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata());
        Map<String, Object> duplicateJsonSchemaLock = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());

        catalog.put("schemaCount", 99);
        lock.put("contracts", List.of("broken"));
        report.put("valid", "true");
        duplicateCatalog.put("contracts", duplicatedContracts(duplicateCatalog));
        duplicateLock.put("contracts", duplicatedContracts(duplicateLock));
        duplicateJsonSchemaCatalog.put(
                "contracts",
                duplicatedJsonSchemaIdContracts(duplicateJsonSchemaCatalog));
        duplicateJsonSchemaLock.put(
                "contracts",
                duplicatedJsonSchemaIdContracts(duplicateJsonSchemaLock));
        mismatchedLock.put("schemaSetSha256", "0000000000000000000000000000000000000000000000000000000000000000");
        Map<String, Object> brokenMismatchReport = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(mismatchedLock));
        Object rawMismatch = ((List<?>) brokenMismatchReport.get("mismatches")).get(0);
        assertTrue(rawMismatch instanceof Map<?, ?>);
        Map<String, Object> brokenMismatch = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) rawMismatch).entrySet()) {
            brokenMismatch.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        brokenMismatch.remove("actual");
        brokenMismatchReport.put("mismatches", List.of(brokenMismatch));

        IllegalArgumentException catalogError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaCatalogSnapshot.fromMetadata(catalog));
        IllegalArgumentException lockError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaLockSnapshot.fromMetadata(lock));
        IllegalArgumentException reportError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(report));
        IllegalArgumentException mismatchError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaLockValidationReport.fromMetadata(brokenMismatchReport));
        IllegalArgumentException duplicateCatalogError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaCatalogSnapshot.fromMetadata(duplicateCatalog));
        IllegalArgumentException duplicateLockError = assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenDatasetSchemaLockSnapshot.fromMetadata(duplicateLock));
        DiscreteTokenDatasetSchemaCatalogSnapshot duplicateJsonSchemaCatalogSnapshot =
                DiscreteTokenDatasetSchemaCatalogSnapshot.fromMetadata(duplicateJsonSchemaCatalog);
        DiscreteTokenDatasetSchemaLockSnapshot duplicateJsonSchemaLockSnapshot =
                DiscreteTokenDatasetSchemaLockSnapshot.fromMetadata(duplicateJsonSchemaLock);
        IllegalArgumentException duplicateCatalogSchemaIdError = assertThrows(
                IllegalArgumentException.class,
                duplicateJsonSchemaCatalogSnapshot::contractsByJsonSchemaId);
        IllegalArgumentException duplicateLockSchemaIdError = assertThrows(
                IllegalArgumentException.class,
                duplicateJsonSchemaLockSnapshot::contractsByJsonSchemaId);

        assertTrue(catalogError.getMessage().contains("schemaCount"));
        assertTrue(lockError.getMessage().contains("contracts must contain objects"));
        assertTrue(reportError.getMessage().contains("valid must be a boolean"));
        assertTrue(mismatchError.getMessage().contains("actual must be present"));
        assertTrue(duplicateCatalogError.getMessage().contains("schema catalog duplicate contract name"));
        assertTrue(duplicateLockError.getMessage().contains("schema lock duplicate contract name"));
        assertTrue(duplicateCatalogSchemaIdError.getMessage().contains("schema catalog duplicate JSON schema id"));
        assertTrue(duplicateLockSchemaIdError.getMessage().contains("schema lock duplicate JSON schema id"));
    }

    @Test
    void exposesLineageHealthSchemaLockJsonSchemaResource() {
        Map<String, Object> schema = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockJsonSchemaMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_JSON_SCHEMA_ID, schema.get("$id"));
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.get("$schema"));
        assertEquals("object", schema.get("type"));
        assertTrue(((List<?>) schema.get("required")).contains("schemaSetSha256"));
        assertTrue(((List<?>) schema.get("required")).contains("lockSchemaId"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("contract"));
    }

    @Test
    void exposesLineageHealthSchemaLockValidationJsonSchemaResource() {
        Map<String, Object> schema =
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockValidationJsonSchemaMetadata();

        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_JSON_SCHEMA_ID,
                schema.get("$id"));
        assertEquals("https://json-schema.org/draft/2020-12/schema", schema.get("$schema"));
        assertEquals("object", schema.get("type"));
        assertTrue(((List<?>) schema.get("required")).contains("valid"));
        assertTrue(((List<?>) schema.get("required")).contains("mismatchCount"));
        assertTrue(((List<?>) schema.get("required")).contains("validationSchemaId"));
        assertTrue(((List<?>) schema.get("required")).contains("actualLockSchemaId"));
        assertTrue(((Map<?, ?>) schema.get("properties")).containsKey("schemaSetMatch"));
        assertTrue(((Map<?, ?>) schema.get("$defs")).containsKey("mismatch"));
    }

    @Test
    void exposesLineageHealthSchemaCatalog() {
        Map<String, Object> compactCatalog = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata();
        Map<String, Object> catalog = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.catalogMetadata(true);
        Map<String, Object> lock = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata();

        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.KIND, catalog.get("kind"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.SCHEMA_VERSION,
                catalog.get("schemaVersion"));
        assertEquals(2, catalog.get("schemaCount"));
        assertEquals(2, compactCatalog.get("schemaCount"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.JSON_SCHEMA_ID,
                catalog.get("catalogSchemaId"));
        assertEquals(64, String.valueOf(catalog.get("catalogSchemaSha256")).length());
        assertTrue(((Number) catalog.get("catalogSchemaByteCount")).intValue() > 0);
        assertEquals(64, String.valueOf(catalog.get("schemaSetSha256")).length());
        assertEquals(catalog.get("schemaSetSha256"), compactCatalog.get("schemaSetSha256"));
        assertEquals(catalog.get("schemaSetSha256"), lock.get("schemaSetSha256"));
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_KIND, lock.get("kind"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_JSON_SCHEMA_ID,
                lock.get("lockSchemaId"));
        assertTrue(((Number) catalog.get("schemaSetByteCount")).intValue()
                > ((Number) catalog.get("catalogSchemaByteCount")).intValue());
        assertEquals(catalog.get("schemaSetByteCount"), lock.get("schemaSetByteCount"));
        assertFalse(((Map<?, ?>) ((List<?>) compactCatalog.get("contracts")).get(0)).containsKey("schema"));
        assertFalse(((Map<?, ?>) ((List<?>) lock.get("contracts")).get(0)).containsKey("schema"));
        List<?> contracts = (List<?>) catalog.get("contracts");
        assertEquals(2, contracts.size());
        Map<?, ?> health = (Map<?, ?>) contracts.get(0);
        Map<?, ?> validation = (Map<?, ?>) contracts.get(1);
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND, health.get("payloadKind"));
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, health.get("jsonSchemaId"));
        assertEquals(64, String.valueOf(health.get("jsonSchemaSha256")).length());
        assertTrue(((Number) health.get("jsonSchemaByteCount")).intValue() > 0);
        assertTrue(((List<?>) health.get("inspectorSections")).contains("lineageHealthSchema"));
        assertTrue(health.containsKey("schema"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.KIND,
                validation.get("payloadKind"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_ID,
                validation.get("jsonSchemaId"));
        assertEquals(64, String.valueOf(validation.get("jsonSchemaSha256")).length());
        assertTrue(((Number) validation.get("jsonSchemaByteCount")).intValue() > 0);
        assertTrue(((List<?>) validation.get("inspectorSections")).contains("lineageHealthValidationSchema"));
        assertTrue(validation.containsKey("schema"));
    }

    @Test
    void validatesCurrentLineageHealthSchemaLock() {
        Map<String, Object> validation =
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateCurrentLock();
        Map<String, Object> lock = DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata();

        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_KIND,
                validation.get("kind"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_SCHEMA_VERSION,
                validation.get("schemaVersion"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_JSON_SCHEMA_ID,
                validation.get("validationSchemaId"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_JSON_SCHEMA_RESOURCE,
                validation.get("validationSchemaResource"));
        assertEquals("valid", validation.get("status"));
        assertTrue((Boolean) validation.get("valid"));
        assertEquals("GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_VALID", validation.get("code"));
        assertEquals(lock.get("schemaSetSha256"), validation.get("expectedSchemaSetSha256"));
        assertEquals(lock.get("schemaSetSha256"), validation.get("actualSchemaSetSha256"));
        assertEquals(lock.get("lockSchemaId"), validation.get("expectedLockSchemaId"));
        assertEquals(lock.get("lockSchemaId"), validation.get("actualLockSchemaId"));
        assertTrue((Boolean) validation.get("schemaSetMatch"));
        assertEquals(0, ((Number) validation.get("mismatchCount")).intValue());
        assertTrue(((List<?>) validation.get("mismatches")).isEmpty());
        assertEquals(
                validation,
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.requireCurrentLockValid());
    }

    @Test
    void reportsLineageHealthSchemaLockMismatch() {
        Map<String, Object> expected = new LinkedHashMap<>(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata());
        expected.put("schemaSetSha256", "0000000000000000000000000000000000000000000000000000000000000000");

        Map<String, Object> validation =
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(expected);

        assertEquals("invalid", validation.get("status"));
        assertFalse((Boolean) validation.get("valid"));
        assertEquals("GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_MISMATCH", validation.get("code"));
        assertFalse((Boolean) validation.get("schemaSetMatch"));
        assertEquals(expected.get("schemaSetSha256"), validation.get("expectedSchemaSetSha256"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata().get("schemaSetSha256"),
                validation.get("actualSchemaSetSha256"));
        assertTrue(((Number) validation.get("mismatchCount")).intValue() > 0);
        assertFalse(((List<?>) validation.get("mismatches")).isEmpty());
        assertThrows(
                IllegalStateException.class,
                () -> DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.requireValidLockMetadata(expected));
    }

    @Test
    void matchesLineageHealthSchemaLockGoldenFixtures() {
        String lockJson = readResource(FIXTURE_ROOT + "lineage-health-schema-lock.v1.json").stripTrailing();
        String validValidationJson = readResource(
                        FIXTURE_ROOT + "lineage-health-schema-lock-validation-valid.v1.json")
                .stripTrailing();
        String mismatchValidationJson = readResource(
                        FIXTURE_ROOT + "lineage-health-schema-lock-validation-mismatch.v1.json")
                .stripTrailing();

        Map<String, Object> lock = DiscreteTokenDatasetCheckpointMetadataJson.fromJson(lockJson);
        Map<String, Object> mismatchedLock = new LinkedHashMap<>(lock);
        mismatchedLock.put("schemaSetSha256", "0000000000000000000000000000000000000000000000000000000000000000");

        assertEquals(
                lockJson,
                DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.lockMetadata()));
        assertEquals(
                validValidationJson,
                DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(lock)));
        assertEquals(
                mismatchValidationJson,
                DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.validateLockMetadata(mismatchedLock)));
    }

    @Test
    void rehydratesHealthyGoldenFixture() {
        Map<String, Object> metadata = readFixture("lineage-health-healthy.v1.json");

        DiscreteTokenDatasetCheckpointLineageHealthSnapshot snapshot =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.fromMetadata(metadata);

        assertMetadataEquals(metadata, snapshot.toMetadata());
        assertTrue(snapshot.healthy());
        assertFalse(snapshot.hasFailures());
        assertEquals("healthy", snapshot.status());
        assertEquals(1.0d, snapshot.healthScore());
        assertTrue(snapshot.primaryIssueCode().isEmpty());
        assertTrue(snapshot.primaryFailingCheckCode().isEmpty());
        snapshot.requireCurrentSchema();
        snapshot.requireHealthy();
    }

    @Test
    void rehydratesMissingParentGoldenFixture() {
        Map<String, Object> metadata = readFixture("lineage-health-missing-parent.v1.json");

        DiscreteTokenDatasetCheckpointLineageHealthSnapshot snapshot =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.fromMetadata(metadata);

        assertMetadataEquals(metadata, snapshot.toMetadata());
        assertFalse(snapshot.healthy());
        assertTrue(snapshot.hasFailures());
        assertEquals("unhealthy", snapshot.status());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", snapshot.primaryIssueCode().orElseThrow());
        assertEquals("missing-parent", snapshot.primaryIssueType().orElseThrow());
        assertEquals("parentExists", snapshot.primaryFailingCheckName().orElseThrow());
        assertEquals("GOLLEK_LINEAGE_MISSING_PARENT", snapshot.primaryFailingCheckCode().orElseThrow());
        assertTrue(snapshot.primaryFailingCheckAction().orElseThrow().contains("Restore the parent checkpoint"));
        assertThrows(IllegalStateException.class, snapshot::requireHealthy);
    }

    @Test
    void validatesGoldenFixtureJsonAndPath(@TempDir Path tempDir) throws IOException {
        String json = readResource(FIXTURE_ROOT + "lineage-health-missing-parent.v1.json");
        Path path = tempDir.resolve("lineage-health.json");
        Files.writeString(path, json, StandardCharsets.UTF_8);

        DiscreteTokenDatasetCheckpointLineageHealthValidationReport jsonReport =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.validateJson(json);
        DiscreteTokenDatasetCheckpointLineageHealthValidationReport pathReport =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.validatePath(path);
        Path reportPath = tempDir.resolve("lineage-health-validation.json");
        Files.writeString(
                reportPath,
                DiscreteTokenDatasetCheckpointMetadataJson.toPrettyJson(jsonReport.toMetadata()),
                StandardCharsets.UTF_8);

        assertTrue(jsonReport.valid());
        assertEquals("valid", jsonReport.status());
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.KIND,
                jsonReport.toMetadata().get("kind"));
        assertEquals(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.SCHEMA_VERSION,
                jsonReport.toMetadata().get("schemaVersion"));
        assertEquals("GOLLEK_LINEAGE_HEALTH_VALID", jsonReport.code());
        assertEquals(DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID, jsonReport.schemaId());
        assertEquals("missing-parent", jsonReport.payloadSummary().get("primaryIssueType"));
        assertMetadataEquals(jsonReport.toMetadata(), pathReport.toMetadata());
        assertMetadataEquals(
                jsonReport.toMetadata(),
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport
                        .fromValidationMetadata(jsonReport.toMetadata())
                        .toMetadata());
        assertMetadataEquals(
                jsonReport.toMetadata(),
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport
                        .fromValidationJson(DiscreteTokenDatasetCheckpointMetadataJson.toJson(jsonReport.toMetadata()))
                        .toMetadata());
        assertMetadataEquals(
                jsonReport.toMetadata(),
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport
                        .fromValidationPath(reportPath)
                        .toMetadata());
        jsonReport.requireValid();
    }

    @Test
    void reportsInvalidPayloadWithoutThrowing() {
        Map<String, Object> broken = new LinkedHashMap<>(readFixture("lineage-health-healthy.v1.json"));
        broken.put("schemaVersion", "gollek.checkpoint-lineage-health.v2");

        DiscreteTokenDatasetCheckpointLineageHealthValidationReport report =
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.validateMetadata(broken);

        assertFalse(report.valid());
        assertEquals("invalid", report.status());
        assertEquals("GOLLEK_LINEAGE_HEALTH_INVALID", report.code());
        assertEquals("gollek.checkpoint-lineage-health.v2", report.payloadSummary().get("schemaVersion"));
        assertTrue(report.errors().get(0).contains("schemaVersion"));
        assertThrows(IllegalStateException.class, report::requireValid);
    }

    private static DiscreteTokenDatasetSchemaBundle lineageHealthSchemaBundle() {
        return lineageHealthSchemaBundleBuilder()
                .addContract(lineageHealthContract())
                .addContract(lineageHealthValidationContract())
                .build();
    }

    private static DiscreteTokenDatasetSchemaBundle.Builder lineageHealthSchemaBundleBuilder() {
        return DiscreteTokenDatasetSchemaBundle
                .builder(DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.class)
                .catalog(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.KIND,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.SCHEMA_VERSION,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.JSON_SCHEMA_ID,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.JSON_SCHEMA_RESOURCE)
                .lock(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_KIND,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_SCHEMA_VERSION,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_JSON_SCHEMA_ID,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_JSON_SCHEMA_RESOURCE)
                .lockValidation(
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_KIND,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_SCHEMA_VERSION,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_JSON_SCHEMA_ID,
                        DiscreteTokenDatasetCheckpointLineageHealthSchemaCatalog.LOCK_VALIDATION_JSON_SCHEMA_RESOURCE)
                .labels(
                        "checkpoint lineage health schema catalog",
                        "checkpoint lineage health schema lock",
                        "checkpoint lineage health schema lock validation")
                .codes(
                        "GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_VALID",
                        "GOLLEK_LINEAGE_HEALTH_SCHEMA_LOCK_MISMATCH");
    }

    private static DiscreteTokenDatasetSchemaContract lineageHealthContract() {
        return new DiscreteTokenDatasetSchemaContract(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.class,
                "checkpointLineageHealth",
                "Checkpoint lineage health",
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_ID,
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.JSON_SCHEMA_RESOURCE,
                List.of("lineageHealth", "lineageHealthSchema"));
    }

    private static DiscreteTokenDatasetSchemaContract lineageHealthValidationContract() {
        return new DiscreteTokenDatasetSchemaContract(
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.class,
                "checkpointLineageHealthValidation",
                "Checkpoint lineage health validation report",
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.KIND,
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.SCHEMA_VERSION,
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_ID,
                DiscreteTokenDatasetCheckpointLineageHealthValidationReport.JSON_SCHEMA_RESOURCE,
                List.of("lineageHealthValidation", "lineageHealthValidationSchema"));
    }

    private static List<Map<String, Object>> duplicatedContracts(Map<String, Object> metadata) {
        Object contracts = metadata.get("contracts");
        if (!(contracts instanceof List<?> list) || list.isEmpty() || !(list.get(0) instanceof Map<?, ?> first)) {
            throw new IllegalArgumentException("metadata must contain at least one contract");
        }
        Map<String, Object> duplicate = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : first.entrySet()) {
            duplicate.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return List.of(duplicate, duplicate);
    }

    private static List<Map<String, Object>> duplicatedJsonSchemaIdContracts(Map<String, Object> metadata) {
        Object contracts = metadata.get("contracts");
        if (!(contracts instanceof List<?> list) || list.size() < 2) {
            throw new IllegalArgumentException("metadata must contain at least two contracts");
        }
        Map<String, Object> first = copiedContract(list.get(0));
        Map<String, Object> second = copiedContract(list.get(1));
        second.put("name", second.get("name") + "Duplicate");
        second.put("jsonSchemaId", first.get("jsonSchemaId"));
        return List.of(first, second);
    }

    private static Map<String, Object> copiedContract(Object contract) {
        if (!(contract instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("contract must be a map");
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            copy.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return copy;
    }

    private static Map<String, Object> readFixture(String name) {
        return DiscreteTokenDatasetCheckpointMetadataJson.fromJson(readResource(FIXTURE_ROOT + name));
    }

    private static void assertMetadataEquals(Map<String, Object> expected, Map<String, Object> actual) {
        assertTrue(
                DiscreteTokenDatasetMetadataSupport.metadataValueMatches(expected, actual, true),
                () -> "expected metadata " + expected + " but was " + actual);
    }

    private static String readResource(String name) {
        try (InputStream stream = DiscreteTokenDatasetCheckpointLineageHealthSnapshotTest.class
                .getClassLoader()
                .getResourceAsStream(name)) {
            if (stream == null) {
                throw new IllegalStateException("test resource not found: " + name);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read test resource: " + name, e);
        }
    }
}
