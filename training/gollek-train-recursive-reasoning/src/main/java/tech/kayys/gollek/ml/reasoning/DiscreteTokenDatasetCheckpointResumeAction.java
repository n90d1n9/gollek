package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Ordered resume preflight action for CI, dashboards, and operator UIs.
 */
public record DiscreteTokenDatasetCheckpointResumeAction(
        int rank,
        String code,
        String hint,
        String gateId,
        String category,
        String status,
        String severity,
        boolean primary,
        boolean attentionRequired,
        boolean actionRequired,
        String gateSummary) {
    public DiscreteTokenDatasetCheckpointResumeAction {
        if (rank < 1) {
            throw new IllegalArgumentException("rank must be positive");
        }
        code = DiscreteTokenDatasetMetadataSupport.requireText(code, "code");
        hint = DiscreteTokenDatasetMetadataSupport.requireText(hint, "hint");
        gateId = DiscreteTokenDatasetMetadataSupport.requireText(gateId, "gateId");
        category = DiscreteTokenDatasetMetadataSupport.requireText(category, "category");
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        severity = DiscreteTokenDatasetMetadataSupport.requireText(severity, "severity");
        gateSummary = DiscreteTokenDatasetMetadataSupport.requireText(gateSummary, "gateSummary");
    }

    public static DiscreteTokenDatasetCheckpointResumeAction fromGate(
            int rank,
            DiscreteTokenDatasetCheckpointResumeGate gate,
            boolean primary) {
        Objects.requireNonNull(gate, "gate must not be null");
        return new DiscreteTokenDatasetCheckpointResumeAction(
                rank,
                gate.actionCode(),
                gate.actionHint(),
                gate.id(),
                gate.category(),
                gate.status(),
                gate.severity(),
                primary,
                gate.attentionRequired(),
                gate.actionRequired(),
                gate.summary());
    }

    public static DiscreteTokenDatasetCheckpointResumeAction fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeAction(
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "rank"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "code"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "hint"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "gateId"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "category"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "severity"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "primary"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "attentionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "actionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "gateSummary"));
    }

    public boolean blocked() {
        return actionRequired;
    }

    public boolean warning() {
        return attentionRequired && !actionRequired;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rank", rank);
        metadata.put("code", code);
        metadata.put("hint", hint);
        metadata.put("gateId", gateId);
        metadata.put("category", category);
        metadata.put("status", status);
        metadata.put("severity", severity);
        metadata.put("primary", primary);
        metadata.put("attentionRequired", attentionRequired);
        metadata.put("actionRequired", actionRequired);
        metadata.put("gateSummary", gateSummary);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }
}
