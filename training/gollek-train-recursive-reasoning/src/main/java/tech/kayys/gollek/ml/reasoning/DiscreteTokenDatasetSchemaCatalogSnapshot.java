package tech.kayys.gollek.ml.reasoning;

import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.immutableStringMap;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredInt;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredMapList;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredString;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredStringList;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed read-side wrapper for schema-catalog metadata.
 */
public record DiscreteTokenDatasetSchemaCatalogSnapshot(Map<String, Object> metadata) {
    public DiscreteTokenDatasetSchemaCatalogSnapshot {
        metadata = immutableStringMap(metadata, "metadata");
        requiredString(metadata, "kind");
        requiredString(metadata, "schemaVersion");
        requiredString(metadata, "catalogSchemaId");
        requiredString(metadata, "catalogSchemaResource");
        requiredString(metadata, "catalogSchemaSha256");
        requiredInt(metadata, "catalogSchemaByteCount");
        requiredString(metadata, "schemaSetSha256");
        requiredInt(metadata, "schemaSetByteCount");
        int schemaCount = requiredInt(metadata, "schemaCount");
        List<Map<String, Object>> contracts = requiredContractList(metadata, "contracts");
        if (schemaCount != contracts.size()) {
            throw new IllegalArgumentException("schemaCount must equal contracts size");
        }
        contractIndex(contracts);
        requiredString(metadata, "summary");
    }

    public static DiscreteTokenDatasetSchemaCatalogSnapshot fromMetadata(Map<?, ?> metadata) {
        return new DiscreteTokenDatasetSchemaCatalogSnapshot(immutableStringMap(metadata, "metadata"));
    }

    public static DiscreteTokenDatasetSchemaCatalogSnapshot fromJson(String json) {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json));
    }

    public static DiscreteTokenDatasetSchemaCatalogSnapshot fromPath(Path path) throws IOException {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.read(path));
    }

    public String kind() {
        return requiredString(metadata, "kind");
    }

    public String schemaVersion() {
        return requiredString(metadata, "schemaVersion");
    }

    public String catalogSchemaId() {
        return requiredString(metadata, "catalogSchemaId");
    }

    public String catalogSchemaResource() {
        return requiredString(metadata, "catalogSchemaResource");
    }

    public String catalogSchemaSha256() {
        return requiredString(metadata, "catalogSchemaSha256");
    }

    public int catalogSchemaByteCount() {
        return requiredInt(metadata, "catalogSchemaByteCount");
    }

    public String schemaSetSha256() {
        return requiredString(metadata, "schemaSetSha256");
    }

    public int schemaSetByteCount() {
        return requiredInt(metadata, "schemaSetByteCount");
    }

    public int schemaCount() {
        return requiredInt(metadata, "schemaCount");
    }

    public DiscreteTokenDatasetSchemaSetIdentity schemaSetIdentity() {
        return DiscreteTokenDatasetSchemaSetIdentity.fromCatalogSnapshot(this);
    }

    public boolean matchesSchemaSet(DiscreteTokenDatasetSchemaSetIdentity expected) {
        return schemaSetIdentity().matches(expected);
    }

    public boolean matchesSchemaSet(DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return matchesSchemaSet(DiscreteTokenDatasetSchemaSetIdentity.fromLockSnapshot(expectedLock));
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireSchemaSet(
            DiscreteTokenDatasetSchemaSetIdentity expected) {
        expected.requireMatches(schemaSetIdentity(), "schema catalog schema set");
        return schemaSetIdentity();
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireSchemaSet(
            DiscreteTokenDatasetSchemaLockSnapshot expectedLock) {
        return requireSchemaSet(DiscreteTokenDatasetSchemaSetIdentity.fromLockSnapshot(expectedLock));
    }

    public List<Map<String, Object>> contracts() {
        return requiredContractList(metadata, "contracts");
    }

    public List<DiscreteTokenDatasetSchemaContractSnapshot> contractDetails() {
        return contractIndex().details();
    }

    public Map<String, DiscreteTokenDatasetSchemaContractSnapshot> contractsByName() {
        return contractIndex().byName();
    }

    public Map<String, DiscreteTokenDatasetSchemaContractSnapshot> contractsByJsonSchemaId() {
        return contractIndex().byJsonSchemaId();
    }

    public Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> contractsByPayloadKind() {
        return contractIndex().byPayloadKind();
    }

    public List<String> contractNames() {
        return contractIndex().names();
    }

    public Optional<Map<String, Object>> contract(String name) {
        return contractDetail(name).map(DiscreteTokenDatasetSchemaContractSnapshot::toMetadata);
    }

    public Map<String, Object> requireContract(String name) {
        return requireContractDetail(name).toMetadata();
    }

    public Optional<DiscreteTokenDatasetSchemaContractSnapshot> contractDetail(String name) {
        return contractIndex().byName(name);
    }

    public DiscreteTokenDatasetSchemaContractSnapshot requireContractDetail(String name) {
        return contractIndex().requireByName(name);
    }

    public Optional<DiscreteTokenDatasetSchemaContractSnapshot> contractForJsonSchemaId(String jsonSchemaId) {
        return contractIndex().byJsonSchemaId(jsonSchemaId);
    }

    public DiscreteTokenDatasetSchemaContractSnapshot requireContractForJsonSchemaId(String jsonSchemaId) {
        return contractIndex().requireByJsonSchemaId(jsonSchemaId);
    }

    public List<DiscreteTokenDatasetSchemaContractSnapshot> contractsForPayloadKind(String payloadKind) {
        return contractIndex().forPayloadKind(payloadKind);
    }

    public Optional<DiscreteTokenDatasetSchemaContractSnapshot> contractForPayloadSchema(
            String payloadKind,
            String payloadSchemaVersion) {
        return contractIndex().forPayloadSchema(payloadKind, payloadSchemaVersion);
    }

    public DiscreteTokenDatasetSchemaContractSnapshot requireContractForPayloadSchema(
            String payloadKind,
            String payloadSchemaVersion) {
        return contractIndex().requireForPayloadSchema(payloadKind, payloadSchemaVersion);
    }

    public boolean hasEmbeddedSchemas() {
        return contractIndex().hasEmbeddedSchemas();
    }

    public String summary() {
        return requiredString(metadata, "summary");
    }

    public boolean isCurrentSchema(
            String expectedKind,
            String expectedSchemaVersion,
            String expectedCatalogSchemaId) {
        return kind().equals(expectedKind)
                && schemaVersion().equals(expectedSchemaVersion)
                && catalogSchemaId().equals(expectedCatalogSchemaId);
    }

    public void requireCurrentSchema(
            String expectedKind,
            String expectedSchemaVersion,
            String expectedCatalogSchemaId) {
        if (!isCurrentSchema(expectedKind, expectedSchemaVersion, expectedCatalogSchemaId)) {
            throw new IllegalStateException("schema catalog is "
                    + kind()
                    + " "
                    + schemaVersion()
                    + " "
                    + catalogSchemaId()
                    + " but expected "
                    + expectedKind
                    + " "
                    + expectedSchemaVersion
                    + " "
                    + expectedCatalogSchemaId);
        }
    }

    public Map<String, Object> toMetadata() {
        return metadata;
    }

    private static List<Map<String, Object>> requiredContractList(Map<String, Object> metadata, String field) {
        List<Map<String, Object>> contracts = requiredMapList(metadata, field);
        for (Map<String, Object> contract : contracts) {
            requiredString(contract, "name");
            requiredString(contract, "label");
            requiredString(contract, "payloadKind");
            requiredString(contract, "payloadSchemaVersion");
            requiredString(contract, "jsonSchemaId");
            requiredString(contract, "jsonSchemaResource");
            requiredString(contract, "jsonSchemaSha256");
            requiredInt(contract, "jsonSchemaByteCount");
            requiredString(contract, "jsonSchemaDraft");
            requiredString(contract, "title");
            requiredString(contract, "description");
            requiredStringList(contract, "inspectorSections");
            if (contract.containsKey("schema") && !(contract.get("schema") instanceof Map<?, ?>)) {
                throw new IllegalArgumentException("schema must be an object when present");
            }
        }
        return contracts;
    }

    private DiscreteTokenDatasetSchemaContractIndex contractIndex() {
        return contractIndex(contracts());
    }

    private static DiscreteTokenDatasetSchemaContractIndex contractIndex(
            List<Map<String, Object>> contracts) {
        return DiscreteTokenDatasetSchemaContractIndex.fromMetadataList(contracts, "schema catalog");
    }
}
