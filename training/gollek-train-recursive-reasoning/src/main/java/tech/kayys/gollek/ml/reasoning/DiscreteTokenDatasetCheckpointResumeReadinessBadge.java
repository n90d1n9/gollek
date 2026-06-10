package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Render-friendly badge for checkpoint-resume preflight state.
 */
public record DiscreteTokenDatasetCheckpointResumeReadinessBadge(
        String status,
        String label,
        String tone,
        String severity,
        String iconKey,
        boolean attentionRequired,
        boolean actionRequired,
        String primaryMessage,
        String actionCode,
        String actionHint,
        String primaryGateId,
        String primaryGateCategory,
    String primaryGateStatus,
        String primaryGateSeverity) {
    public DiscreteTokenDatasetCheckpointResumeReadinessBadge {
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        label = DiscreteTokenDatasetMetadataSupport.requireText(label, "label");
        tone = DiscreteTokenDatasetMetadataSupport.requireText(tone, "tone");
        severity = DiscreteTokenDatasetMetadataSupport.requireText(severity, "severity");
        iconKey = DiscreteTokenDatasetMetadataSupport.requireText(iconKey, "iconKey");
        primaryMessage = DiscreteTokenDatasetMetadataSupport.requireText(primaryMessage, "primaryMessage");
        actionCode = DiscreteTokenDatasetMetadataSupport.requireText(actionCode, "actionCode");
        actionHint = DiscreteTokenDatasetMetadataSupport.requireText(actionHint, "actionHint");
        primaryGateId = DiscreteTokenDatasetMetadataSupport.optionalText(primaryGateId, "primaryGateId");
        primaryGateCategory =
                DiscreteTokenDatasetMetadataSupport.optionalText(primaryGateCategory, "primaryGateCategory");
        primaryGateStatus = DiscreteTokenDatasetMetadataSupport.optionalText(primaryGateStatus, "primaryGateStatus");
        primaryGateSeverity =
                DiscreteTokenDatasetMetadataSupport.optionalText(primaryGateSeverity, "primaryGateSeverity");
        verifyConsistency(status, label, tone, iconKey, attentionRequired, actionRequired);
    }

    public static DiscreteTokenDatasetCheckpointResumeReadinessBadge fromSummary(
            DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        Objects.requireNonNull(summary, "summary must not be null");
        Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryGate = summary.primaryGate();
        return new DiscreteTokenDatasetCheckpointResumeReadinessBadge(
                summary.status(),
                label(summary),
                tone(summary),
                summary.highestSeverity(),
                iconKey(summary),
                summary.attentionRequired(),
                summary.actionRequired(),
                primaryMessage(summary, primaryGate),
                actionCode(summary),
                actionHint(summary),
                primaryGate.map(DiscreteTokenDatasetCheckpointResumeGate::id).orElse(null),
                primaryGate.map(DiscreteTokenDatasetCheckpointResumeGate::category).orElse(null),
                primaryGate.map(DiscreteTokenDatasetCheckpointResumeGate::status).orElse(null),
                primaryGate.map(DiscreteTokenDatasetCheckpointResumeGate::severity).orElse(null));
    }

    public static DiscreteTokenDatasetCheckpointResumeReadinessBadge fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeReadinessBadge(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "label"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "tone"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "severity"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "iconKey"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "attentionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "actionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "primaryMessage"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "actionCode"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "actionHint"),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "primaryGateId"),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "primaryGateCategory"),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "primaryGateStatus"),
                DiscreteTokenDatasetMetadataSupport.optionalString(metadata, "primaryGateSeverity"));
    }

    public boolean ready() {
        return "accepted".equals(status) && !attentionRequired && !actionRequired;
    }

    public boolean blocked() {
        return actionRequired;
    }

    public boolean warning() {
        return attentionRequired && !actionRequired;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status);
        metadata.put("label", label);
        metadata.put("tone", tone);
        metadata.put("severity", severity);
        metadata.put("iconKey", iconKey);
        metadata.put("attentionRequired", attentionRequired);
        metadata.put("actionRequired", actionRequired);
        metadata.put("primaryMessage", primaryMessage);
        metadata.put("actionCode", actionCode);
        metadata.put("actionHint", actionHint);
        if (primaryGateId != null) {
            metadata.put("primaryGateId", primaryGateId);
        }
        if (primaryGateCategory != null) {
            metadata.put("primaryGateCategory", primaryGateCategory);
        }
        if (primaryGateStatus != null) {
            metadata.put("primaryGateStatus", primaryGateStatus);
        }
        if (primaryGateSeverity != null) {
            metadata.put("primaryGateSeverity", primaryGateSeverity);
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static String label(DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        return switch (summary.status()) {
            case "blocked" -> "Blocked";
            case "warning" -> "Warning";
            default -> "Ready";
        };
    }

    private static String tone(DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        return switch (summary.status()) {
            case "blocked" -> "danger";
            case "warning" -> "warning";
            default -> "success";
        };
    }

    private static String iconKey(DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        return switch (summary.status()) {
            case "blocked" -> "x-circle";
            case "warning" -> "alert-triangle";
            default -> "check-circle";
        };
    }

    private static String primaryMessage(
            DiscreteTokenDatasetCheckpointResumeGateSummary summary,
            Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryGate) {
        if (summary.actionRequired() || summary.attentionRequired()) {
            return primaryGate
                    .map(DiscreteTokenDatasetCheckpointResumeGate::summary)
                    .orElse(summary.summary());
        }
        return summary.summary();
    }

    private static String actionHint(DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        return summary.primaryActionHint()
                .orElse("Continue training from the selected checkpoint.");
    }

    private static String actionCode(DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        return summary.primaryActionCode()
                .orElse("continue");
    }

    private static void verifyConsistency(
            String status,
            String label,
            String tone,
            String iconKey,
            boolean attentionRequired,
            boolean actionRequired) {
        if (actionRequired && !attentionRequired) {
            throw new IllegalArgumentException("actionRequired badges must also require attention");
        }
        switch (status) {
            case "blocked" -> {
                requireBadgeShape(status, label, tone, iconKey, "Blocked", "danger", "x-circle");
                if (!actionRequired) {
                    throw new IllegalArgumentException("blocked badges must require action");
                }
            }
            case "warning" -> {
                requireBadgeShape(status, label, tone, iconKey, "Warning", "warning", "alert-triangle");
                if (!attentionRequired || actionRequired) {
                    throw new IllegalArgumentException("warning badges must require attention but not action");
                }
            }
            case "accepted" -> {
                requireBadgeShape(status, label, tone, iconKey, "Ready", "success", "check-circle");
                if (attentionRequired || actionRequired) {
                    throw new IllegalArgumentException("accepted badges must not require attention or action");
                }
            }
            default -> throw new IllegalArgumentException("status must be accepted, warning, or blocked");
        }
    }

    private static void requireBadgeShape(
            String status,
            String label,
            String tone,
            String iconKey,
            String expectedLabel,
            String expectedTone,
            String expectedIconKey) {
        if (!expectedLabel.equals(label)) {
            throw new IllegalArgumentException("label does not match " + status + " badge status");
        }
        if (!expectedTone.equals(tone)) {
            throw new IllegalArgumentException("tone does not match " + status + " badge status");
        }
        if (!expectedIconKey.equals(iconKey)) {
            throw new IllegalArgumentException("iconKey does not match " + status + " badge status");
        }
    }

}
