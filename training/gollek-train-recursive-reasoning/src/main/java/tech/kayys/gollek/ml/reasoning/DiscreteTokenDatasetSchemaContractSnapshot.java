package tech.kayys.gollek.ml.reasoning;

import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.immutableStringMap;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.optionalString;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.optionalStringList;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredInt;
import static tech.kayys.gollek.ml.reasoning.DiscreteTokenDatasetSchemaMetadata.requiredString;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Typed read-side wrapper for one schema-catalog or schema-lock contract entry.
 */
public record DiscreteTokenDatasetSchemaContractSnapshot(Map<String, Object> metadata) {
    public DiscreteTokenDatasetSchemaContractSnapshot {
        metadata = immutableStringMap(metadata, "metadata");
        requiredString(metadata, "name");
        requiredString(metadata, "payloadKind");
        requiredString(metadata, "payloadSchemaVersion");
        requiredString(metadata, "jsonSchemaId");
        requiredString(metadata, "jsonSchemaSha256");
        requiredInt(metadata, "jsonSchemaByteCount");
        optionalString(metadata, "label");
        optionalString(metadata, "jsonSchemaResource");
        optionalString(metadata, "jsonSchemaDraft");
        optionalString(metadata, "title");
        optionalString(metadata, "description");
        optionalStringList(metadata, "inspectorSections");
        if (metadata.containsKey("schema") && !(metadata.get("schema") instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("schema must be an object when present");
        }
    }

    public static DiscreteTokenDatasetSchemaContractSnapshot fromMetadata(Map<?, ?> metadata) {
        return new DiscreteTokenDatasetSchemaContractSnapshot(immutableStringMap(metadata, "metadata"));
    }

    static List<DiscreteTokenDatasetSchemaContractSnapshot> fromMetadataList(List<Map<String, Object>> contracts) {
        return contracts.stream()
                .map(DiscreteTokenDatasetSchemaContractSnapshot::fromMetadata)
                .toList();
    }

    static Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byName(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String owner) {
        Objects.requireNonNull(contracts, "contracts must not be null");
        owner = requiredOwner(owner);
        Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byName = new LinkedHashMap<>();
        for (DiscreteTokenDatasetSchemaContractSnapshot contract : contracts) {
            DiscreteTokenDatasetSchemaContractSnapshot previous = byName.putIfAbsent(contract.name(), contract);
            if (previous != null) {
                throw new IllegalArgumentException(owner + " duplicate contract name: " + contract.name());
            }
        }
        return Collections.unmodifiableMap(byName);
    }

    static DiscreteTokenDatasetSchemaContractSnapshot requireByName(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String name,
            String owner) {
        Objects.requireNonNull(name, "name must not be null");
        owner = requiredOwner(owner);
        DiscreteTokenDatasetSchemaContractSnapshot contract = byName(contracts, owner).get(name);
        if (contract == null) {
            throw new IllegalArgumentException(owner + " contract not found: " + name);
        }
        return contract;
    }

    static Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byJsonSchemaId(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String owner) {
        Objects.requireNonNull(contracts, "contracts must not be null");
        owner = requiredOwner(owner);
        Map<String, DiscreteTokenDatasetSchemaContractSnapshot> bySchemaId = new LinkedHashMap<>();
        for (DiscreteTokenDatasetSchemaContractSnapshot contract : contracts) {
            DiscreteTokenDatasetSchemaContractSnapshot previous =
                    bySchemaId.putIfAbsent(contract.jsonSchemaId(), contract);
            if (previous != null) {
                throw new IllegalArgumentException(owner + " duplicate JSON schema id: " + contract.jsonSchemaId());
            }
        }
        return Collections.unmodifiableMap(bySchemaId);
    }

    static DiscreteTokenDatasetSchemaContractSnapshot requireByJsonSchemaId(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String jsonSchemaId,
            String owner) {
        Objects.requireNonNull(jsonSchemaId, "jsonSchemaId must not be null");
        owner = requiredOwner(owner);
        DiscreteTokenDatasetSchemaContractSnapshot contract = byJsonSchemaId(contracts, owner).get(jsonSchemaId);
        if (contract == null) {
            throw new IllegalArgumentException(owner + " contract not found for JSON schema id: " + jsonSchemaId);
        }
        return contract;
    }

    static Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> byPayloadKind(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts) {
        Objects.requireNonNull(contracts, "contracts must not be null");
        Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> grouped = new LinkedHashMap<>();
        for (DiscreteTokenDatasetSchemaContractSnapshot contract : contracts) {
            grouped.computeIfAbsent(contract.payloadKind(), ignored -> new ArrayList<>()).add(contract);
        }
        Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> immutable = new LinkedHashMap<>();
        grouped.forEach((kind, values) -> immutable.put(kind, List.copyOf(values)));
        return Collections.unmodifiableMap(immutable);
    }

    static List<DiscreteTokenDatasetSchemaContractSnapshot> forPayloadKind(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String payloadKind) {
        Objects.requireNonNull(payloadKind, "payloadKind must not be null");
        return byPayloadKind(contracts).getOrDefault(payloadKind, List.of());
    }

    static Optional<DiscreteTokenDatasetSchemaContractSnapshot> findPayloadSchema(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String payloadKind,
            String payloadSchemaVersion) {
        Objects.requireNonNull(payloadKind, "payloadKind must not be null");
        Objects.requireNonNull(payloadSchemaVersion, "payloadSchemaVersion must not be null");
        return contracts.stream()
                .filter(contract -> contract.isPayloadSchema(payloadKind, payloadSchemaVersion))
                .findFirst();
    }

    static DiscreteTokenDatasetSchemaContractSnapshot requirePayloadSchema(
            List<DiscreteTokenDatasetSchemaContractSnapshot> contracts,
            String payloadKind,
            String payloadSchemaVersion,
            String owner) {
        String validatedOwner = requiredOwner(owner);
        return findPayloadSchema(contracts, payloadKind, payloadSchemaVersion)
                .orElseThrow(() -> new IllegalArgumentException(validatedOwner
                        + " contract not found for payload schema: "
                        + payloadKind
                        + " "
                        + payloadSchemaVersion));
    }

    public String name() {
        return requiredString(metadata, "name");
    }

    public Optional<String> label() {
        return optionalString(metadata, "label");
    }

    public String payloadKind() {
        return requiredString(metadata, "payloadKind");
    }

    public String payloadSchemaVersion() {
        return requiredString(metadata, "payloadSchemaVersion");
    }

    public boolean isPayloadSchema(String expectedPayloadKind, String expectedPayloadSchemaVersion) {
        return payloadKind().equals(expectedPayloadKind)
                && payloadSchemaVersion().equals(expectedPayloadSchemaVersion);
    }

    public String jsonSchemaId() {
        return requiredString(metadata, "jsonSchemaId");
    }

    public boolean isJsonSchema(String expectedJsonSchemaId) {
        Objects.requireNonNull(expectedJsonSchemaId, "expectedJsonSchemaId must not be null");
        return jsonSchemaId().equals(expectedJsonSchemaId);
    }

    public Optional<String> jsonSchemaResource() {
        return optionalString(metadata, "jsonSchemaResource");
    }

    public String jsonSchemaSha256() {
        return requiredString(metadata, "jsonSchemaSha256");
    }

    public int jsonSchemaByteCount() {
        return requiredInt(metadata, "jsonSchemaByteCount");
    }

    public Optional<String> jsonSchemaDraft() {
        return optionalString(metadata, "jsonSchemaDraft");
    }

    public Optional<String> title() {
        return optionalString(metadata, "title");
    }

    public Optional<String> description() {
        return optionalString(metadata, "description");
    }

    public List<String> inspectorSections() {
        return optionalStringList(metadata, "inspectorSections");
    }

    public boolean hasEmbeddedSchema() {
        return metadata.containsKey("schema");
    }

    public Optional<Map<String, Object>> schemaMetadata() {
        Object value = metadata.get("schema");
        if (value instanceof Map<?, ?> map) {
            return Optional.of(immutableStringMap(map, "schema"));
        }
        return Optional.empty();
    }

    public Map<String, Object> toMetadata() {
        return metadata;
    }

    private static String requiredOwner(String owner) {
        owner = Objects.requireNonNull(owner, "owner must not be null");
        if (owner.isBlank()) {
            throw new IllegalArgumentException("owner must not be blank");
        }
        return owner;
    }
}
