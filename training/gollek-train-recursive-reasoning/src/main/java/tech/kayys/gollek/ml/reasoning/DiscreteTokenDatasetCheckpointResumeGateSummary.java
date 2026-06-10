package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

/**
 * Compact aggregate of checkpoint-resume preflight gates.
 */
public record DiscreteTokenDatasetCheckpointResumeGateSummary(
        List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
    public DiscreteTokenDatasetCheckpointResumeGateSummary {
        gates = immutableGates(gates);
        verifyUniqueGateIds(gates);
    }

    public static DiscreteTokenDatasetCheckpointResumeGateSummary fromGates(
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        return new DiscreteTokenDatasetCheckpointResumeGateSummary(gates);
    }

    public static DiscreteTokenDatasetCheckpointResumeGateSummary fromMetadata(
            Map<?, ?> metadata,
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        return fromGates(gates).requireMetadataMatch(metadata);
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

    public boolean actionRequired() {
        return blockedCount() > 0;
    }

    public boolean attentionRequired() {
        return actionRequired() || warningCount() > 0;
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

    public String highestSeverity() {
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

    public Map<String, List<DiscreteTokenDatasetCheckpointResumeGate>> gatesByStatus() {
        return groupGatesBy(DiscreteTokenDatasetCheckpointResumeGate::status);
    }

    public Map<String, List<DiscreteTokenDatasetCheckpointResumeGate>> gatesBySeverity() {
        return groupGatesBy(DiscreteTokenDatasetCheckpointResumeGate::severity);
    }

    public Map<String, List<DiscreteTokenDatasetCheckpointResumeGate>> gatesByCategory() {
        return groupGatesBy(DiscreteTokenDatasetCheckpointResumeGate::category);
    }

    public Map<String, List<String>> gateIdsByStatus() {
        return groupGateIdsBy(DiscreteTokenDatasetCheckpointResumeGate::status);
    }

    public Map<String, List<String>> gateIdsBySeverity() {
        return groupGateIdsBy(DiscreteTokenDatasetCheckpointResumeGate::severity);
    }

    public Map<String, List<String>> gateIdsByCategory() {
        return groupGateIdsBy(DiscreteTokenDatasetCheckpointResumeGate::category);
    }

    public Map<String, Integer> gateCountByStatus() {
        return countGateIds(gateIdsByStatus());
    }

    public Map<String, Integer> gateCountBySeverity() {
        return countGateIds(gateIdsBySeverity());
    }

    public Map<String, Integer> gateCountByCategory() {
        return countGateIds(gateIdsByCategory());
    }

    public List<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> gateCategorySummaries() {
        return gatesByCategory().entrySet().stream()
                .map(entry -> DiscreteTokenDatasetCheckpointResumeGateCategorySummary.fromGates(
                        entry.getKey(),
                        entry.getValue()))
                .toList();
    }

    public Map<String, DiscreteTokenDatasetCheckpointResumeGateCategorySummary> gateCategorySummaryByCategory() {
        Map<String, DiscreteTokenDatasetCheckpointResumeGateCategorySummary> summaries = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGateCategorySummary summary : gateCategorySummaries()) {
            summaries.put(summary.category(), summary);
        }
        return Collections.unmodifiableMap(summaries);
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> gateCategorySummary(String category) {
        return Optional.ofNullable(gateCategorySummaryByCategory().get(
                DiscreteTokenDatasetMetadataSupport.requireText(category, "category")));
    }

    public DiscreteTokenDatasetCheckpointResumeGateCategorySummary requireGateCategorySummary(String category) {
        return gateCategorySummary(category)
                .orElseThrow(() -> new IllegalArgumentException("unknown checkpoint resume gate category: " + category));
    }

    public List<String> categoryIds() {
        return gateCategorySummaries().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category)
                .toList();
    }

    public List<String> acceptedCategories() {
        return acceptedCategorySummaries().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category)
                .toList();
    }

    public List<String> blockedCategories() {
        return blockedCategorySummaries().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category)
                .toList();
    }

    public List<String> warningCategories() {
        return warningCategorySummaries().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> acceptedCategorySummaries() {
        return gateCategorySummaries().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::allAccepted)
                .filter(summary -> summary.warningCount() == 0)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> blockedCategorySummaries() {
        return gateCategorySummaries().stream()
                .filter(summary -> summary.blockedCount() > 0)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> warningCategorySummaries() {
        return gateCategorySummaries().stream()
                .filter(summary -> summary.warningCount() > 0)
                .toList();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> primaryBlockedCategorySummary() {
        return blockedCategorySummaries().stream().findFirst();
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeGateCategorySummary> primaryWarningCategorySummary() {
        return warningCategorySummaries().stream().findFirst();
    }

    public Optional<String> primaryBlockedCategory() {
        return primaryBlockedCategorySummary()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category);
    }

    public Optional<String> primaryWarningCategory() {
        return primaryWarningCategorySummary()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::category);
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

    public Optional<DiscreteTokenDatasetCheckpointResumeGate> primaryGate() {
        return primaryBlockedGate()
                .or(this::primaryWarningGate)
                .or(() -> gates.stream().findFirst());
    }

    public Optional<String> primaryGateId() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::id);
    }

    public Optional<String> primaryGateCategory() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::category);
    }

    public Optional<String> primaryGateStatus() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::status);
    }

    public Optional<String> primaryGateSeverity() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::severity);
    }

    public Optional<String> primaryActionCode() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::actionCode);
    }

    public Optional<String> primaryActionHint() {
        return primaryGate().map(DiscreteTokenDatasetCheckpointResumeGate::actionHint);
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> nextActions() {
        List<DiscreteTokenDatasetCheckpointResumeGate> actionGates;
        if (attentionRequired()) {
            List<DiscreteTokenDatasetCheckpointResumeGate> ordered = new ArrayList<>();
            ordered.addAll(gates.stream()
                    .filter(DiscreteTokenDatasetCheckpointResumeGate::blocked)
                    .toList());
            ordered.addAll(gates.stream()
                    .filter(DiscreteTokenDatasetCheckpointResumeGate::warning)
                    .toList());
            actionGates = ordered;
        } else {
            actionGates = primaryGate().stream().toList();
        }
        String primaryGateId = primaryGateId().orElse(null);
        List<DiscreteTokenDatasetCheckpointResumeAction> actions = new ArrayList<>();
        for (int i = 0; i < actionGates.size(); i++) {
            DiscreteTokenDatasetCheckpointResumeGate gate = actionGates.get(i);
            actions.add(DiscreteTokenDatasetCheckpointResumeAction.fromGate(
                    i + 1,
                    gate,
                    gate.id().equals(primaryGateId)));
        }
        return List.copyOf(actions);
    }

    public Optional<DiscreteTokenDatasetCheckpointResumeAction> primaryAction() {
        return nextActions().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::primary)
                .findFirst()
                .or(() -> nextActions().stream().findFirst());
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> requiredActions() {
        return nextActions().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::blocked)
                .toList();
    }

    public List<DiscreteTokenDatasetCheckpointResumeAction> warningActions() {
        return nextActions().stream()
                .filter(DiscreteTokenDatasetCheckpointResumeAction::warning)
                .toList();
    }

    public List<String> actionCodes() {
        return nextActions().stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
    }

    public List<String> requiredActionCodes() {
        return requiredActions().stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
    }

    public List<String> warningActionCodes() {
        return warningActions().stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::code)
                .toList();
    }

    public DiscreteTokenDatasetCheckpointResumeActionPlan actionPlan() {
        return DiscreteTokenDatasetCheckpointResumeActionPlan.fromSummary(this);
    }

    public DiscreteTokenDatasetCheckpointResumeReadinessBadge readinessBadge() {
        return DiscreteTokenDatasetCheckpointResumeReadinessBadge.fromSummary(this);
    }

    public DiscreteTokenDatasetCheckpointResumeGateSummary requireAllAccepted() {
        if (!allAccepted()) {
            throw new IllegalStateException(summary());
        }
        return this;
    }

    public DiscreteTokenDatasetCheckpointResumeGateSummary requireMetadataMatch(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        Map<String, Object> expected = toMetadata();
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            if (isOptionalDerivedIndex(entry.getKey())
                    && (!metadata.containsKey(entry.getKey()) || metadata.get(entry.getKey()) == null)) {
                continue;
            }
            Object actual = DiscreteTokenDatasetMetadataSupport.required(metadata, entry.getKey(), "gateSummary");
            if (!gateSummaryMetadataMatches(entry.getValue(), actual)) {
                throw new IllegalArgumentException(
                        "gateSummary field '" + entry.getKey() + "' does not match resume gates");
            }
        }
        rejectUnexpectedOptional(metadata, expected, "primaryBlockedGateId");
        rejectUnexpectedOptional(metadata, expected, "primaryWarningGateId");
        rejectUnexpectedOptional(metadata, expected, "primaryBlockedCategory");
        rejectUnexpectedOptional(metadata, expected, "primaryWarningCategory");
        return this;
    }

    public String summary() {
        if (blockedCount() > 0) {
            return "checkpoint resume gates blocked: "
                    + String.join(", ", blockedGateIds());
        }
        if (warningCount() > 0) {
            return "checkpoint resume gates accepted with warning(s): "
                    + String.join(", ", warningGateIds());
        }
        return "checkpoint resume gates accepted: " + acceptedCount() + "/" + gateCount();
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("status", status());
        metadata.put("allAccepted", allAccepted());
        metadata.put("attentionRequired", attentionRequired());
        metadata.put("actionRequired", actionRequired());
        metadata.put("highestSeverity", highestSeverity());
        primaryActionCode().ifPresent(actionCode -> metadata.put("primaryActionCode", actionCode));
        primaryActionHint().ifPresent(actionHint -> metadata.put("primaryActionHint", actionHint));
        metadata.put("nextActions", nextActions().stream()
                .map(DiscreteTokenDatasetCheckpointResumeAction::toMetadata)
                .toList());
        metadata.put("actionCodes", actionCodes());
        metadata.put("requiredActionCodes", requiredActionCodes());
        metadata.put("warningActionCodes", warningActionCodes());
        metadata.put("actionPlan", actionPlan().toMetadata());
        metadata.put("readinessBadge", readinessBadge().toMetadata());
        metadata.put("gateCount", gateCount());
        metadata.put("acceptedCount", acceptedCount());
        metadata.put("blockedCount", blockedCount());
        metadata.put("warningCount", warningCount());
        metadata.put("gateIds", gateIds());
        metadata.put("acceptedGateIds", acceptedGateIds());
        metadata.put("blockedGateIds", blockedGateIds());
        metadata.put("warningGateIds", warningGateIds());
        metadata.put("gateIdsByStatus", gateIdsByStatus());
        metadata.put("gateIdsBySeverity", gateIdsBySeverity());
        metadata.put("gateIdsByCategory", gateIdsByCategory());
        metadata.put("gateCountByStatus", gateCountByStatus());
        metadata.put("gateCountBySeverity", gateCountBySeverity());
        metadata.put("gateCountByCategory", gateCountByCategory());
        metadata.put("gateCategorySummaries", gateCategorySummaries().stream()
                .map(DiscreteTokenDatasetCheckpointResumeGateCategorySummary::toMetadata)
                .toList());
        metadata.put("categoryIds", categoryIds());
        metadata.put("acceptedCategories", acceptedCategories());
        metadata.put("blockedCategories", blockedCategories());
        metadata.put("warningCategories", warningCategories());
        primaryBlockedGate().ifPresent(gate -> metadata.put("primaryBlockedGateId", gate.id()));
        primaryWarningGate().ifPresent(gate -> metadata.put("primaryWarningGateId", gate.id()));
        primaryGateId().ifPresent(id -> metadata.put("primaryGateId", id));
        primaryGateCategory().ifPresent(category -> metadata.put("primaryGateCategory", category));
        primaryGateStatus().ifPresent(status -> metadata.put("primaryGateStatus", status));
        primaryGateSeverity().ifPresent(severity -> metadata.put("primaryGateSeverity", severity));
        primaryBlockedCategory().ifPresent(category -> metadata.put("primaryBlockedCategory", category));
        primaryWarningCategory().ifPresent(category -> metadata.put("primaryWarningCategory", category));
        metadata.put("summary", summary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private Map<String, List<DiscreteTokenDatasetCheckpointResumeGate>> groupGatesBy(
            Function<DiscreteTokenDatasetCheckpointResumeGate, String> classifier) {
        Map<String, List<DiscreteTokenDatasetCheckpointResumeGate>> grouped = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates) {
            grouped.computeIfAbsent(classifier.apply(gate), ignored -> new ArrayList<>()).add(gate);
        }
        return immutableListMap(grouped);
    }

    private static Map<String, Integer> countGateIds(Map<String, List<String>> gateIds) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : gateIds.entrySet()) {
            counts.put(entry.getKey(), entry.getValue().size());
        }
        return Collections.unmodifiableMap(counts);
    }

    private Map<String, List<String>> groupGateIdsBy(
            Function<DiscreteTokenDatasetCheckpointResumeGate, String> classifier) {
        Map<String, List<String>> grouped = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates) {
            grouped.computeIfAbsent(classifier.apply(gate), ignored -> new ArrayList<>()).add(gate.id());
        }
        return immutableListMap(grouped);
    }

    private static List<DiscreteTokenDatasetCheckpointResumeGate> immutableGates(
            List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        if (gates == null || gates.isEmpty()) {
            throw new IllegalArgumentException("gates must not be empty");
        }
        return gates.stream()
                .map(gate -> Objects.requireNonNull(gate, "gates entries must not be null"))
                .toList();
    }

    private static void verifyUniqueGateIds(List<DiscreteTokenDatasetCheckpointResumeGate> gates) {
        Map<String, Boolean> ids = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointResumeGate gate : gates) {
            if (ids.put(gate.id(), true) != null) {
                throw new IllegalArgumentException("duplicate resume gate id: " + gate.id());
            }
        }
    }

    private static void rejectUnexpectedOptional(Map<?, ?> metadata, Map<String, Object> expected, String key) {
        if (!expected.containsKey(key) && metadata.containsKey(key) && metadata.get(key) != null) {
            throw new IllegalArgumentException("gateSummary field '" + key + "' does not match resume gates");
        }
    }

    private static boolean isOptionalDerivedIndex(String key) {
        return "gateIdsByStatus".equals(key)
                || "gateIdsBySeverity".equals(key)
                || "gateIdsByCategory".equals(key)
                || "gateCountByStatus".equals(key)
                || "gateCountBySeverity".equals(key)
                || "gateCountByCategory".equals(key)
                || "gateCategorySummaries".equals(key)
                || "categoryIds".equals(key)
                || "acceptedCategories".equals(key)
                || "blockedCategories".equals(key)
                || "warningCategories".equals(key)
                || "primaryBlockedCategory".equals(key)
                || "primaryWarningCategory".equals(key)
                || "attentionRequired".equals(key)
                || "actionRequired".equals(key)
                || "highestSeverity".equals(key)
                || "primaryGateId".equals(key)
                || "primaryGateCategory".equals(key)
                || "primaryGateStatus".equals(key)
                || "primaryGateSeverity".equals(key)
                || "primaryActionCode".equals(key)
                || "primaryActionHint".equals(key)
                || "nextActions".equals(key)
                || "actionCodes".equals(key)
                || "requiredActionCodes".equals(key)
                || "warningActionCodes".equals(key)
                || "actionPlan".equals(key)
                || "readinessBadge".equals(key);
    }

    private static boolean gateSummaryMetadataMatches(Object expected, Object actual) {
        return DiscreteTokenDatasetMetadataSupport.metadataValueMatches(
                expected,
                actual,
                true,
                DiscreteTokenDatasetCheckpointResumeGateSummary::isOptionalNestedDerivedField,
                DiscreteTokenDatasetCheckpointResumeGateSummary::legacyActionHintMatches);
    }

    private static boolean isOptionalNestedDerivedField(Object key) {
        return key instanceof CharSequence text
                && ("attentionRequired".contentEquals(text)
                        || "actionRequired".contentEquals(text)
                        || "actionCode".contentEquals(text)
                        || "primaryActionCode".contentEquals(text)
                        || "primaryActionHint".contentEquals(text));
    }

    private static boolean legacyActionHintMatches(Object key, Object expected, Object actual) {
        if (!(key instanceof CharSequence text) || !"actionHint".contentEquals(text)) {
            return false;
        }
        if (!(expected instanceof CharSequence expectedText) || !(actual instanceof CharSequence actualText)) {
            return false;
        }
        String expectedHint = expectedText.toString();
        String actualHint = actualText.toString();
        return ("continue".equals(actualHint)
                        && "Continue training from the selected checkpoint.".equals(expectedHint))
                || ("resolve-blocked-resume-gate".equals(actualHint)
                        && !"Continue training from the selected checkpoint.".equals(expectedHint)
                        && !expectedHint.startsWith("Review "))
                || ("review-warning-resume-gate".equals(actualHint)
                        && expectedHint.startsWith("Review "));
    }

    private static <T> Map<String, List<T>> immutableListMap(Map<String, List<T>> values) {
        Map<String, List<T>> copy = new LinkedHashMap<>();
        for (Map.Entry<String, List<T>> entry : values.entrySet()) {
            copy.put(entry.getKey(), List.copyOf(entry.getValue()));
        }
        return Collections.unmodifiableMap(copy);
    }

}
