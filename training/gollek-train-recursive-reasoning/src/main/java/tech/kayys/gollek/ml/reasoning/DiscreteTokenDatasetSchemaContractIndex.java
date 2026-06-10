package tech.kayys.gollek.ml.reasoning;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

final class DiscreteTokenDatasetSchemaContractIndex {
    private final String owner;
    private final List<DiscreteTokenDatasetSchemaContractSnapshot> details;
    private final Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byName;
    private final Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> byPayloadKind;

    private DiscreteTokenDatasetSchemaContractIndex(
            String owner,
            List<DiscreteTokenDatasetSchemaContractSnapshot> details) {
        this.owner = requiredOwner(owner);
        this.details = List.copyOf(Objects.requireNonNull(details, "details must not be null"));
        this.byName = DiscreteTokenDatasetSchemaContractSnapshot.byName(this.details, this.owner);
        this.byPayloadKind = DiscreteTokenDatasetSchemaContractSnapshot.byPayloadKind(this.details);
    }

    static DiscreteTokenDatasetSchemaContractIndex fromMetadataList(
            List<Map<String, Object>> contracts,
            String owner) {
        return new DiscreteTokenDatasetSchemaContractIndex(
                owner,
                DiscreteTokenDatasetSchemaContractSnapshot.fromMetadataList(contracts));
    }

    List<DiscreteTokenDatasetSchemaContractSnapshot> details() {
        return details;
    }

    Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byName() {
        return byName;
    }

    Map<String, DiscreteTokenDatasetSchemaContractSnapshot> byJsonSchemaId() {
        return DiscreteTokenDatasetSchemaContractSnapshot.byJsonSchemaId(details, owner);
    }

    Map<String, List<DiscreteTokenDatasetSchemaContractSnapshot>> byPayloadKind() {
        return byPayloadKind;
    }

    List<String> names() {
        return details.stream()
                .map(DiscreteTokenDatasetSchemaContractSnapshot::name)
                .toList();
    }

    Optional<DiscreteTokenDatasetSchemaContractSnapshot> byName(String name) {
        Objects.requireNonNull(name, "name must not be null");
        return Optional.ofNullable(byName.get(name));
    }

    DiscreteTokenDatasetSchemaContractSnapshot requireByName(String name) {
        return DiscreteTokenDatasetSchemaContractSnapshot.requireByName(details, name, owner);
    }

    Optional<DiscreteTokenDatasetSchemaContractSnapshot> byJsonSchemaId(String jsonSchemaId) {
        Objects.requireNonNull(jsonSchemaId, "jsonSchemaId must not be null");
        return Optional.ofNullable(byJsonSchemaId().get(jsonSchemaId));
    }

    DiscreteTokenDatasetSchemaContractSnapshot requireByJsonSchemaId(String jsonSchemaId) {
        return DiscreteTokenDatasetSchemaContractSnapshot.requireByJsonSchemaId(details, jsonSchemaId, owner);
    }

    List<DiscreteTokenDatasetSchemaContractSnapshot> forPayloadKind(String payloadKind) {
        return DiscreteTokenDatasetSchemaContractSnapshot.forPayloadKind(details, payloadKind);
    }

    Optional<DiscreteTokenDatasetSchemaContractSnapshot> forPayloadSchema(
            String payloadKind,
            String payloadSchemaVersion) {
        return DiscreteTokenDatasetSchemaContractSnapshot.findPayloadSchema(
                details,
                payloadKind,
                payloadSchemaVersion);
    }

    DiscreteTokenDatasetSchemaContractSnapshot requireForPayloadSchema(
            String payloadKind,
            String payloadSchemaVersion) {
        return DiscreteTokenDatasetSchemaContractSnapshot.requirePayloadSchema(
                details,
                payloadKind,
                payloadSchemaVersion,
                owner);
    }

    boolean hasEmbeddedSchemas() {
        return details.stream().anyMatch(DiscreteTokenDatasetSchemaContractSnapshot::hasEmbeddedSchema);
    }

    private static String requiredOwner(String owner) {
        owner = Objects.requireNonNull(owner, "owner must not be null");
        if (owner.isBlank()) {
            throw new IllegalArgumentException("owner must not be blank");
        }
        return owner;
    }
}
