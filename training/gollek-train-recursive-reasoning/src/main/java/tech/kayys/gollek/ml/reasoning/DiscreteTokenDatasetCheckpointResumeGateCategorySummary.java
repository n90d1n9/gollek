package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Per-category rollup of checkpoint-resume preflight gates.
 */
public record DiscreteTokenDatasetCheckpointResumeGateCategorySummary(
        String category,
        List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
    public DiscreteTokenDatasetCheckpointResumeGateCategorySummary {
        category = DiscreteTokenDatasetMetadataSupport.requireText(category, "category");
        gates = immutableGates(category, gates);
    }

    public static DiscreteTokenDatasetCheckpointResumeGateCategorySummary fromGates(
            String category,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        return new DiscreteTokenDatasetCheckpointResumeGateCategorySummary(category, gates);
    }

    public static DiscreteTokenDatasetCheckpointResumeGateCategorySummary fromMetadata(
            Map<?, ?> metadata,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        String category = DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "category");
        return fromGates(category, matchingCategoryGates(category, gates)).requireMetadataMatch(metadata);
    }

    public int gateCount() {
        return gates.size();
    }

    public int acceptedCount() {
        return acceptedGateIds().size();
    }

    public int blockedCount() {
        return blockedGateIds().size();
    }

    public int warningCount() {
        return warningGateIds().size();
    }

    public boolean allAccepted() {
        return blockedCount() == 0;
    }

    public boolean attentionRequired() {
        return blockedCount() > 0 || warningCount() > 0;
    }

    public boolean actionRequired() {
        return blockedCount() > 0;
    }

    public String status() {
        if (blockedCount() > 0) {
            return "blocked";
        }
        if (warningCount() > 0) {
            return "warning";
        }
        return "accepted";
    }

    public String severity() {
        if (gates.stream().anyMatch(gate -> "error".equals(gate.severity()))) {
            return "error";
        }
        if (gates.stream().anyMatch(gate -> "warning".equals(gate.severity()))) {
            return "warning";
        }
        return "info";
    }

    public List<String> gateIds() {
        return gates.stream()
                .map(DiscreteTokenDatasetCheckpointResumeGate::id)
                .toList();
    }

    public List<String> acceptedGateIds() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::accepted)
                .map(DiscreteTokenDatasetCheckpointResumeGate::id)
                .toList();
    }

    public List<String> blockedGateIds() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::blocked)
                .map(DiscreteTokenDatasetCheckpointResumeGate::id)
                .toList();
    }

    public List<String> warningGateIds() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::warning)
                .map(DiscreteTokenDatasetCheckpointResumeGate::id)
                .toList();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryGate() {
        return primaryBlockedGate()
                .or(this::primaryWarningGate)
                .or(() -> gates.stream().findFirst());
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryBlockedGate() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::blocked)
                .findFirst();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryWarningGate() {
        return gates.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGate::warning)
                .findFirst();
    }

    public Optional<String> primaryActionCode() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::actionCode);
    }

    public Optional<String> primaryActionHint() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::actionHint);
    }

    public String summary() {
        if (blockedCount() > 0) {
            return "checkpoint resume category " + category + " blocked: "
                    + String.join(", ", blockedGateIds());
        }
        if (warningCount() > 0) {
            return "checkpoint resume category " + category + " accepted with warning(s): "
                    + String.join(", ", warningGateIds());
        }
        return "checkpoint resume category " + category + " accepted: " + acceptedCount() + "/" + gateCount();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("category", category);
        metadata.put("status", status());
        metadata.put("severity", severity());
        metadata.put("allAccepted", allAccepted());
        metadata.put("attentionRequired", attentionRequired());
        metadata.put("actionRequired", actionRequired());
        metadata.put("gateCount", gateCount());
        metadata.put("acceptedCount", acceptedCount());
        metadata.put("blockedCount", blockedCount());
        metadata.put("warningCount", warningCount());
        metadata.put("gateIds", gateIds());
        metadata.put("acceptedGateIds", acceptedGateIds());
        metadata.put("blockedGateIds", blockedGateIds());
        metadata.put("warningGateIds", warningGateIds());
        primaryGate().ifPresent(gate -> metadata.put("primaryGateId", gate.id()));
        primaryActionCode().ifPresent(actionCode -> metadata.put("primaryActionCode", actionCode));
        primaryActionHint().ifPresent(actionHint -> metadata.put("primaryActionHint", actionHint));
        primaryBlockedGate().ifPresent(gate -> metadata.put("primaryBlockedGateId", gate.id()));
        primaryWarningGate().ifPresent(gate -> metadata.put("primaryWarningGateId", gate.id()));
        metadata.put("summary", summary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public DiscreteTokenDatasetCheckpointResumeGateCategorySummary requireMetadataMatch(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Map<String, Object> expected = toMetadata();
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            Object actual = DiscreteTokenDatasetMetadataSupport.required(metadata, entry.getKey());
            if (!DiscreteTokenDatasetMetadataSupport.metadataValueMatches(entry.getValue(), actual)) {
                throw new IllegalArgumentException(
                        "gate category summary field '" + entry.getKey() + "' does not match resume gates");
            }
        }
        rejectUnexpectedOptional(metadata, expected, "primaryBlockedGateId");
        rejectUnexpectedOptional(metadata, expected, "primaryWarningGateId");
        return this;
    }

    private static List<DiscreteTokenDatasetCheckpointResumeGate> immutableGates(
            String category,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        if (gates == null || gates.isEmpty()) {
            throw new IllegalArgumentException("gates must not be empty");
        }
        Map<String, Boolean> ids = new LinkedHashMap<>();
        return gates.stream()
                .map(gate -> {
                    gate = Objects.requireNonNull(gate, "gates entries must not be null");
                    if (!category.equals(gate.category())) {
                        throw new IllegalArgumentException("gate category mismatch: " + gate.id());
                    }
                    if (ids.put(gate.id(), true) != null) {
                        throw new IllegalArgumentException("duplicate resume gate id: " + gate.id());
                    }
                    return gate;
                })
                .toList();
    }

    private static List<DiscreteTokenDatasetCheckpointResumeGate> matchingCategoryGates(
            String category,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        Objects.requireNonNull(gates, "gates must not be null");
        List<DiscreteTokenDatasetCheckpointResumeGate> matching = gates.stream()
                .map(gate -> Objects.requireNonNull(gate, "gates entries must not be null"))
                .filter(gate -> category.equals(gate.category()))
                .toList();
        if (matching.isEmpty()) {
            throw new IllegalArgumentException("no resume gates found for category: " + category);
        }
        return matching;
    }

    private static void rejectUnexpectedOptional(Map<?, ?> metadata, Map<String, Object> expected, String key) {
        if (!expected.containsKey(key) && metadata.containsKey(key) && metadata.get(key) != null) {
            throw new IllegalArgumentException(
                    "gate category summary field '" + key + "' does not match resume gates");
        }
    }

}
