package tech.kayys.gollek.ml.reasoning;

import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.immutableStringMap;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredInt;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredMapList;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredString;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed read-side wrapper for schema-lock metadata.
 */
public record DiscreteTokenDatasetSchemaLockSnapshot(Map<String, Object> metadata) {
    public DiscreteTokenDatasetSchemaLockSnapshot {
        metadata = immutableStringMap(metadata, "metadata");
        requiredString(metadata, "kind");
        requiredString(metadata, "schemaVersion");
        requiredString(metadata, "schemaSetSha256");
        requiredInt(metadata, "schemaSetByteCount");
        int schemaCount = requiredInt(metadata, "schemaCount");
        requiredString(metadata, "catalogSchemaId");
        requiredString(metadata, "catalogSchemaSha256");
        requiredString(metadata, "lockSchemaId");
        List<Map<String, Object>> contracts = requiredContractList(metadata, "contracts");
        if (schemaCount != contracts.size()) {
            throw new IllegalArgumentException("schemaCount must equal contracts size");
        }
        contractIndex(contracts);
        requiredString(metadata, "summary");
    }

    public static DiscreteTokenDatasetSchemaLockSnapshot fromMetadata(Map<?, ?> metadata) {
        return new DiscreteTokenDatasetSchemaLockSnapshot(immutableStringMap(metadata, "metadata"));
    }

    public static DiscreteTokenDatasetSchemaLockSnapshot fromJson(String json) {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.fromJson(json));
    }

    public static DiscreteTokenDatasetSchemaLockSnapshot fromPath(Path path) throws IOException {
        return fromMetadata(DiscreteTokenDatasetCheckpointMetadataJson.read(path));
    }

    public String kind() {
        return requiredString(metadata, "kind");
    }

    public String schemaVersion() {
        return requiredString(metadata, "schemaVersion");
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
        return DiscreteTokenDatasetSchemaSetIdentity.fromLockSnapshot(this);
    }

    public boolean matchesSchemaSet(DiscreteTokenDatasetSchemaSetIdentity expected) {
        return schemaSetIdentity().matches(expected);
    }

    public boolean matchesSchemaSet(DiscreteTokenDatasetSchemaCatalogSnapshot expectedCatalog) {
        return matchesSchemaSet(DiscreteTokenDatasetSchemaSetIdentity.fromCatalogSnapshot(expectedCatalog));
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireSchemaSet(
            DiscreteTokenDatasetSchemaSetIdentity expected) {
        expected.requireMatches(schemaSetIdentity(), "schema lock schema set");
        return schemaSetIdentity();
    }

    public DiscreteTokenDatasetSchemaSetIdentity requireSchemaSet(
            DiscreteTokenDatasetSchemaCatalogSnapshot expectedCatalog) {
        return requireSchemaSet(DiscreteTokenDatasetSchemaSetIdentity.fromCatalogSnapshot(expectedCatalog));
    }

    public String catalogSchemaId() {
        return requiredString(metadata, "catalogSchemaId");
    }

    public String catalogSchemaSha256() {
        return requiredString(metadata, "catalogSchemaSha256");
    }

    public String lockSchemaId() {
        return requiredString(metadata, "lockSchemaId");
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

    public String summary() {
        return requiredString(metadata, "summary");
    }

    public boolean isCurrentSchema(String expectedKind, String expectedSchemaVersion, String expectedLockSchemaId) {
        return kind().equals(expectedKind)
                && schemaVersion().equals(expectedSchemaVersion)
                && lockSchemaId().equals(expectedLockSchemaId);
    }

    public void requireCurrentSchema(String expectedKind, String expectedSchemaVersion, String expectedLockSchemaId) {
        if (!isCurrentSchema(expectedKind, expectedSchemaVersion, expectedLockSchemaId)) {
            throw new IllegalStateException("schema lock is "
                    + kind()
                    + " "
                    + schemaVersion()
                    + " "
                    + lockSchemaId()
                    + " but expected "
                    + expectedKind
                    + " "
                    + expectedSchemaVersion
                    + " "
                    + expectedLockSchemaId);
        }
    }

    public Map<String, Object> toMetadata() {
        return metadata;
    }

    private static List<Map<String, Object>> requiredContractList(Map<String, Object> metadata, String field) {
        List<Map<String, Object>> contracts = requiredMapList(metadata, field);
        for (Map<String, Object> contract : contracts) {
            requiredString(contract, "name");
            requiredString(contract, "payloadKind");
            requiredString(contract, "payloadSchemaVersion");
            requiredString(contract, "jsonSchemaId");
            requiredString(contract, "jsonSchemaSha256");
            requiredInt(contract, "jsonSchemaByteCount");
        }
        return contracts;
    }

    private DiscreteTokenDatasetSchemaContractIndex contractIndex() {
        return contractIndex(contracts());
    }

    private static DiscreteTokenDatasetSchemaContractIndex contractIndex(
            List<Map<String, Object>> contracts) {
        return DiscreteTokenDatasetSchemaContractIndex.fromMetadataList(contracts, "schema lock");
    }
}
