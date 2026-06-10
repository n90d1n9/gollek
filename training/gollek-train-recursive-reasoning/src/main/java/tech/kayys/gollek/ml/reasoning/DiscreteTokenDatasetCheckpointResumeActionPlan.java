package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Single resume preflight action payload for CI, dashboards, and extensions.
 */
public record DiscreteTokenDatasetCheckpointResumeActionPlan(
        String status,
        boolean ready,
        boolean readyWithoutWarnings,
        boolean attentionRequired,
        boolean actionRequired,
        int actionCount,
        int requiredActionCount,
        int warningActionCount,
        String primaryActionCode,
        String primaryActionHint,
        String primaryGateId,
        String primaryCategory,
        List<String> actionCodes,
        List<String> requiredActionCodes,
        List<String> warningActionCodes,
        List<DiscreteTokenDatasetCheckpointResumeAction> actions,
        String summary) {
    public DiscreteTokenDatasetCheckpointResumeActionPlan {
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        primaryActionCode = DiscreteTokenDatasetMetadataSupport.requireText(primaryActionCode, "primaryActionCode");
        primaryActionHint = DiscreteTokenDatasetMetadataSupport.requireText(primaryActionHint, "primaryActionHint");
        primaryGateId = DiscreteTokenDatasetMetadataSupport.requireText(primaryGateId, "primaryGateId");
        primaryCategory = DiscreteTokenDatasetMetadataSupport.requireText(primaryCategory, "primaryCategory");
        actionCodes = DiscreteTokenDatasetMetadataSupport.immutableTextList(actionCodes, "actionCodes");
        requiredActionCodes =
                DiscreteTokenDatasetMetadataSupport.immutableTextList(requiredActionCodes, "requiredActionCodes");
        warningActionCodes =
                DiscreteTokenDatasetMetadataSupport.immutableTextList(warningActionCodes, "warningActionCodes");
        actions = immutableActions(actions);
        summary = DiscreteTokenDatasetMetadataSupport.requireText(summary, "summary");
        requireCount(actionCount, "actionCount");
        requireCount(requiredActionCount, "requiredActionCount");
        requireCount(warningActionCount, "warningActionCount");
        if (actionCount != actions.size()) {
            throw new IllegalArgumentException("actionCount must match actions size");
        }
        if (requiredActionCount != requiredActionCodes.size()) {
            throw new IllegalArgumentException("requiredActionCount must match requiredActionCodes size");
        }
        if (warningActionCount != warningActionCodes.size()) {
            throw new IllegalArgumentException("warningActionCount must match warningActionCodes size");
        }
        verifyConsistency(
                ready,
                readyWithoutWarnings,
                attentionRequired,
                actionRequired,
                actionCodes,
                requiredActionCodes,
                warningActionCodes,
                actions,
                primaryActionCode,
                primaryActionHint,
                primaryGateId,
                primaryCategory,
                requiredActionCount,
                warningActionCount);
    }

    public static DiscreteTokenDatasetCheckpointResumeActionPlan fromSummary(
            DiscreteTokenDatasetCheckpointResumeGateSummary summary) {
        Objects.requireNonNull(summary, "summary must not be null");
        List<DiscreteTokenDatasetCheckpointResumeAction> actions = summary.nextActions();
        DiscreteTokenDatasetCheckpointResumeAction primaryAction = summary.primaryAction()
                .orElseThrow(() -> new IllegalStateException("resume action plan requires at least one action"));
        return new DiscreteTokenDatasetCheckpointResumeActionPlan(
                summary.status(),
                !summary.actionRequired(),
                !summary.attentionRequired(),
                summary.attentionRequired(),
                summary.actionRequired(),
                actions.size(),
                summary.requiredActions().size(),
                summary.warningActions().size(),
                primaryAction.code(),
                primaryAction.hint(),
                primaryAction.gateId(),
                primaryAction.category(),
                summary.actionCodes(),
                summary.requiredActionCodes(),
                summary.warningActionCodes(),
                actions,
                planSummary(summary.status(), summary.requiredActionCodes(), summary.warningActionCodes()));
    }

    public static DiscreteTokenDatasetCheckpointResumeActionPlan fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointResumeActionPlan(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "ready"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "readyWithoutWarnings"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "attentionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "actionRequired"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "actionCount"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "requiredActionCount"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "warningActionCount"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "primaryActionCode"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "primaryActionHint"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "primaryGateId"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "primaryCategory"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "actionCodes"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "requiredActionCodes"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "warningActionCodes"),
                requiredActions(metadata, "actions"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "summary"));
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions() {
        return actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::blocked)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> warningActions() {
        return actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::warning)
                .toList();
    }

    public DiscreteTokenDatasetCheckpointResumeAction primaryAction() {
        return actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::primary)
                .findFirst()
                .orElse(actions.getFirst());
    }

    public DiscreteTokenDatasetCheckpointResumeActionPlan requireNoRequiredActions() {
        if (actionRequired) {
            throw new IllegalStateException(summary);
        }
        return this;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status);
        metadata.put("ready", ready);
        metadata.put("readyWithoutWarnings", readyWithoutWarnings);
        metadata.put("attentionRequired", attentionRequired);
        metadata.put("actionRequired", actionRequired);
        metadata.put("actionCount", actionCount);
        metadata.put("requiredActionCount", requiredActionCount);
        metadata.put("warningActionCount", warningActionCount);
        metadata.put("primaryActionCode", primaryActionCode);
        metadata.put("primaryActionHint", primaryActionHint);
        metadata.put("primaryGateId", primaryGateId);
        metadata.put("primaryCategory", primaryCategory);
        metadata.put("actionCodes", actionCodes);
        metadata.put("requiredActionCodes", requiredActionCodes);
        metadata.put("warningActionCodes", warningActionCodes);
        metadata.put("actions", actions.stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::toMetadata)
                .toList());
        metadata.put("summary", summary);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static String planSummary(
            String status,
            List<String> requiredActionCodes,
            List<String> warningActionCodes) {
        if (!requiredActionCodes.isEmpty()) {
            return "resume action plan blocked: " + String.join(", ", requiredActionCodes);
        }
        if (!warningActionCodes.isEmpty()) {
            return "resume action plan accepted with warning(s): " + String.join(", ", warningActionCodes);
        }
        return "resume action plan " + status + ": continue";
    }

    private static void verifyConsistency(
            boolean ready,
            boolean readyWithoutWarnings,
            boolean attentionRequired,
            boolean actionRequired,
            List<String> actionCodes,
            List<String> requiredActionCodes,
            List<String> warningActionCodes,
            List<DiscreteTokenDatasetCheckpointResumeAction> actions,
            String primaryActionCode,
            String primaryActionHint,
            String primaryGateId,
            String primaryCategory,
            int requiredActionCount,
            int warningActionCount) {
        List<String> actualActionCodes = actions.stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
        if (!actionCodes.equals(actualActionCodes)) {
            throw new IllegalArgumentException("actionCodes must match actions");
        }
        List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions = actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::blocked)
                .toList();
        List<String> actualRequiredActionCodes = requiredActions.stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
        if (requiredActionCount != requiredActions.size()
                || !requiredActionCodes.equals(actualRequiredActionCodes)) {
            throw new IllegalArgumentException("requiredActionCodes must match blocked actions");
        }
        List<DiscreteTokenDatasetCheckpointResumeAction> warningActions = actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::warning)
                .toList();
        List<String> actualWarningActionCodes = warningActions.stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
        if (warningActionCount != warningActions.size()
                || !warningActionCodes.equals(actualWarningActionCodes)) {
            throw new IllegalArgumentException("warningActionCodes must match warning actions");
        }
        long primaryCount = actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::primary)
                .count();
        if (primaryCount != 1L) {
            throw new IllegalArgumentException("actions must contain exactly one primary action");
        }
        DiscreteTokenDatasetCheckpointResumeAction primaryAction = actions.stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::primary)
                .findFirst()
                .orElseThrow();
        if (!primaryActionCode.equals(primaryAction.code())
                || !primaryActionHint.equals(primaryAction.hint())
                || !primaryGateId.equals(primaryAction.gateId())
                || !primaryCategory.equals(primaryAction.category())) {
            throw new IllegalArgumentException("primary action fields must match the primary action");
        }
        if (ready != !actionRequired) {
            throw new IllegalArgumentException("ready must be true when no required action is present");
        }
        if (readyWithoutWarnings != !attentionRequired) {
            throw new IllegalArgumentException("readyWithoutWarnings must be true when no attention is required");
        }
        if (actionRequired != !requiredActions.isEmpty()) {
            throw new IllegalArgumentException("actionRequired must match blocked actions");
        }
        if (attentionRequired != (actionRequired || !warningActions.isEmpty())) {
            throw new IllegalArgumentException("attentionRequired must match blocked or warning actions");
        }
    }

    private static List<DiscreteTokenDatasetCheckpointResumeAction> immutableActions(
            List<DiscreteTokenDatasetCheckpointResumeAction> actions) {
        if (actions == null || actions.isEmpty()) {
            throw new IllegalArgumentException("actions must not be empty");
        }
        return actions.stream()
                .map(action -> Objects.requireNonNull(action, "actions entries must not be null"))
                .toList();
    }

    private static List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions(Map<?, ?> metadata, String key) {
        Object value = DiscreteTokenDatasetMetadataSupport.required(metadata, key);
        if (value instanceof List<?> list) {
            return list.stream()
                    .map(entry -> {
                        if (entry instanceof Map<?, ?> map) {
                            return DiscreteTokenDatasetCheckpointResumeAction.fromMetadata(map);
                        }
                        throw new IllegalArgumentException("metadata field '" + key + "' entries must be maps");
                    })
                    .toList();
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    private static void requireCount(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must not be negative");
        }
    }

}
