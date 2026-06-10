package tech.kayys.gollek.ml.reasoning;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Read-side view of checkpoint lineage health metadata.
 */
public record DiscreteTokenDatasetCheckpointLineageHealthSnapshot(
        String kind,
        String schemaVersion,
        String rootDir,
        String status,
        double healthScore,
        String alertLevel,
        boolean healthy,
        int issueCount,
        List<String> issueTypes,
        int issueDetailCount,
        List<String> issueCodes,
        int blockingIssueCount,
        Map<String, Object> primaryIssue,
        List<Map<String, Object>> issueDetails,
        List<String> duplicateRunIds,
        List<String> missingParentRunIds,
        List<String> ambiguousParentRunIds,
        List<String> parentMismatchRunIds,
        List<String> cycleRunIds,
        List<String> unresolvedRunIds,
        String summaryMessage,
        String recommendedAction,
        Map<String, Object> healthBadge,
        List<Map<String, Object>> checks,
        List<Map<String, Object>> failingChecks,
        Map<String, Object> primaryFailingCheck,
        int passingCheckCount,
        int failingCheckCount,
        Map<String, Object> checkSummary,
        String summary) {

    public static final String KIND = "checkpoint-lineage-health";
    public static final String SCHEMA_VERSION = "gollek.checkpoint-lineage-health.v1";
    public static final String JSON_SCHEMA_ID =
            "https://gollek.ai/schemas/training/checkpoint-lineage-health.v1.schema.json";
    public static final String JSON_SCHEMA_RESOURCE =
            "tech/kayys/gollek/ml/reasoning/schemas/checkpoint-lineage-health.v1.schema.json";

    public DiscreteTokenDatasetCheckpointLineageHealthSnapshot {
        kind = DiscreteTokenDatasetMetadataSupport.requireText(kind, "kind");
        schemaVersion = DiscreteTokenDatasetMetadataSupport.requireText(schemaVersion, "schemaVersion");
        rootDir = DiscreteTokenDatasetMetadataSupport.requireText(rootDir, "rootDir");
        status = DiscreteTokenDatasetMetadataSupport.requireText(status, "status");
        alertLevel = DiscreteTokenDatasetMetadataSupport.requireText(alertLevel, "alertLevel");
        issueTypes = immutableStringList(issueTypes, "issueTypes");
        issueCodes = immutableStringList(issueCodes, "issueCodes");
        primaryIssue = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(primaryIssue, "primaryIssue");
        issueDetails = DiscreteTokenDatasetMetadataSupport.immutableMetadataMapList(issueDetails, "issueDetails");
        duplicateRunIds = immutableStringList(duplicateRunIds, "duplicateRunIds");
        missingParentRunIds = immutableStringList(missingParentRunIds, "missingParentRunIds");
        ambiguousParentRunIds = immutableStringList(ambiguousParentRunIds, "ambiguousParentRunIds");
        parentMismatchRunIds = immutableStringList(parentMismatchRunIds, "parentMismatchRunIds");
        cycleRunIds = immutableStringList(cycleRunIds, "cycleRunIds");
        unresolvedRunIds = immutableStringList(unresolvedRunIds, "unresolvedRunIds");
        summaryMessage = DiscreteTokenDatasetMetadataSupport.requireText(summaryMessage, "summaryMessage");
        recommendedAction = DiscreteTokenDatasetMetadataSupport.requireText(recommendedAction, "recommendedAction");
        healthBadge = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(healthBadge, "healthBadge");
        checks = DiscreteTokenDatasetMetadataSupport.immutableMetadataMapList(checks, "checks");
        failingChecks = DiscreteTokenDatasetMetadataSupport.immutableMetadataMapList(failingChecks, "failingChecks");
        primaryFailingCheck =
                DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(primaryFailingCheck, "primaryFailingCheck");
        checkSummary = DiscreteTokenDatasetMetadataSupport.immutableMetadataMap(checkSummary, "checkSummary");
        summary = DiscreteTokenDatasetMetadataSupport.requireText(summary, "summary");
        verifyConsistency(
                kind,
                schemaVersion,
                status,
                healthScore,
                healthy,
                issueCount,
                issueTypes,
                issueDetailCount,
                issueDetails,
                blockingIssueCount,
                primaryIssue,
                checks,
                failingChecks,
                primaryFailingCheck,
                passingCheckCount,
                failingCheckCount,
                healthBadge,
                checkSummary);
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthSnapshot fromGraph(
            DiscreteTokenDatasetCheckpointLineageGraph graph) {
        Objects.requireNonNull(graph, "graph must not be null");
        return fromMetadata(graph.healthMetadata());
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthSnapshot fromMetadata(Map<?, ?> metadata) {
        Objects.requireNonNull(metadata, "metadata must not be null");
        return new DiscreteTokenDatasetCheckpointLineageHealthSnapshot(
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "kind"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "schemaVersion"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "rootDir"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "status"),
                DiscreteTokenDatasetMetadataSupport.requiredDouble(metadata, "healthScore"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "alertLevel"),
                DiscreteTokenDatasetMetadataSupport.requiredBoolean(metadata, "healthy"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "issueCount"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "issueTypes"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "issueDetailCount"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "issueCodes"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "blockingIssueCount"),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "primaryIssue"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMapList(metadata, "issueDetails"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "duplicateRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "missingParentRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "ambiguousParentRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "parentMismatchRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "cycleRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredStringList(metadata, "unresolvedRunIds"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "summaryMessage"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "recommendedAction"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "healthBadge"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMapList(metadata, "checks"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMapList(metadata, "failingChecks"),
                DiscreteTokenDatasetMetadataSupport.optionalMetadataMap(metadata, "primaryFailingCheck"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "passingCheckCount"),
                DiscreteTokenDatasetMetadataSupport.requiredInt(metadata, "failingCheckCount"),
                DiscreteTokenDatasetMetadataSupport.requiredMetadataMap(metadata, "checkSummary"),
                DiscreteTokenDatasetMetadataSupport.requiredString(metadata, "summary"));
    }

    public static String jsonSchemaText() {
        return DiscreteTokenDatasetSchemaContract.resourceText(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.class,
                JSON_SCHEMA_RESOURCE,
                "checkpoint lineage health JSON schema");
    }

    public static Map<String, Object> jsonSchemaMetadata() {
        return DiscreteTokenDatasetSchemaContract.jsonMetadata(
                DiscreteTokenDatasetCheckpointLineageHealthSnapshot.class,
                JSON_SCHEMA_RESOURCE,
                "checkpoint lineage health JSON schema");
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport validateMetadata(Map<?, ?> metadata) {
        return DiscreteTokenDatasetCheckpointLineageHealthValidationReport.fromMetadata(metadata);
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport validateJson(String json) {
        return DiscreteTokenDatasetCheckpointLineageHealthValidationReport.fromJson(json);
    }

    public static DiscreteTokenDatasetCheckpointLineageHealthValidationReport validatePath(Path path) {
        return DiscreteTokenDatasetCheckpointLineageHealthValidationReport.fromPath(path);
    }

    public boolean isCurrentSchema() {
        return SCHEMA_VERSION.equals(schemaVersion);
    }

    public void requireCurrentSchema() {
        if (!isCurrentSchema()) {
            throw new IllegalStateException(
                    "checkpoint lineage health schema is " + schemaVersion + " but expected " + SCHEMA_VERSION);
        }
    }

    public void requireHealthy() {
        if (!healthy) {
            throw new IllegalStateException(summary);
        }
    }

    public Optional<String> primaryIssueCode() {
        return optionalString(primaryIssue, "code");
    }

    public Optional<String> primaryIssueType() {
        return optionalString(primaryIssue, "type");
    }

    public Optional<String> primaryIssueAction() {
        return optionalString(primaryIssue, "action");
    }

    public Optional<String> primaryFailingCheckName() {
        return optionalString(primaryFailingCheck, "name");
    }

    public Optional<String> primaryFailingCheckType() {
        return optionalString(primaryFailingCheck, "type");
    }

    public Optional<String> primaryFailingCheckCode() {
        return optionalString(primaryFailingCheck, "code");
    }

    public Optional<String> primaryFailingCheckSeverity() {
        return optionalString(primaryFailingCheck, "severity");
    }

    public Optional<String> primaryFailingCheckAction() {
        return optionalString(primaryFailingCheck, "action");
    }

    public Optional<String> primaryFailingCheckMessage() {
        return optionalString(primaryFailingCheck, "detail");
    }

    public boolean hasFailures() {
        return failingCheckCount > 0 || !healthy;
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", kind);
        metadata.put("schemaVersion", schemaVersion);
        metadata.put("rootDir", rootDir);
        metadata.put("status", status);
        metadata.put("healthScore", healthScore);
        metadata.put("alertLevel", alertLevel);
        metadata.put("healthy", healthy);
        metadata.put("issueCount", issueCount);
        metadata.put("issueTypes", issueTypes);
        metadata.put("issueDetailCount", issueDetailCount);
        metadata.put("issueCodes", issueCodes);
        metadata.put("blockingIssueCount", blockingIssueCount);
        if (!primaryIssue.isEmpty()) {
            primaryIssueCode().ifPresent(code -> metadata.put("primaryIssueCode", code));
            primaryIssueType().ifPresent(type -> metadata.put("primaryIssueType", type));
            primaryIssueAction().ifPresent(action -> metadata.put("primaryIssueAction", action));
            metadata.put("primaryIssue", primaryIssue);
        }
        metadata.put("issueDetails", issueDetails);
        metadata.put("duplicateRunIds", duplicateRunIds);
        metadata.put("missingParentRunIds", missingParentRunIds);
        metadata.put("ambiguousParentRunIds", ambiguousParentRunIds);
        metadata.put("parentMismatchRunIds", parentMismatchRunIds);
        metadata.put("cycleRunIds", cycleRunIds);
        metadata.put("unresolvedRunIds", unresolvedRunIds);
        metadata.put("summaryMessage", summaryMessage);
        metadata.put("recommendedAction", recommendedAction);
        metadata.put("healthBadge", healthBadge);
        metadata.put("checks", checks);
        metadata.put("failingChecks", failingChecks);
        if (!primaryFailingCheck.isEmpty()) {
            metadata.put("primaryFailingCheck", primaryFailingCheck);
            primaryFailingCheckName().ifPresent(name -> metadata.put("primaryFailingCheckName", name));
            primaryFailingCheckType().ifPresent(type -> metadata.put("primaryFailingCheckType", type));
            primaryFailingCheckCode().ifPresent(code -> metadata.put("primaryFailingCheckCode", code));
            primaryFailingCheckSeverity().ifPresent(severity -> metadata.put("primaryFailingCheckSeverity", severity));
            primaryFailingCheckAction().ifPresent(action -> metadata.put("primaryFailingCheckAction", action));
            primaryFailingCheckMessage().ifPresent(message -> metadata.put("primaryFailingCheckMessage", message));
        }
        metadata.put("passingCheckCount", passingCheckCount);
        metadata.put("failingCheckCount", failingCheckCount);
        metadata.put("checkSummary", checkSummary);
        metadata.put("summary", summary);
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private static void verifyConsistency(
            String kind,
            String schemaVersion,
            String status,
            double healthScore,
            boolean healthy,
            int issueCount,
            List<String> issueTypes,
            int issueDetailCount,
            List<Map<String, Object>> issueDetails,
            int blockingIssueCount,
            Map<String, Object> primaryIssue,
            List<Map<String, Object>> checks,
            List<Map<String, Object>> failingChecks,
            Map<String, Object> primaryFailingCheck,
            int passingCheckCount,
            int failingCheckCount,
            Map<String, Object> healthBadge,
            Map<String, Object> checkSummary) {
        if (!KIND.equals(kind)) {
            throw new IllegalArgumentException("kind must be " + KIND);
        }
        if (!SCHEMA_VERSION.equals(schemaVersion)) {
            throw new IllegalArgumentException("schemaVersion must be " + SCHEMA_VERSION);
        }
        String expectedStatus = healthy ? "healthy" : "unhealthy";
        if (!expectedStatus.equals(status)) {
            throw new IllegalArgumentException("status must be " + expectedStatus + " when healthy=" + healthy);
        }
        if (!Double.isFinite(healthScore) || healthScore < 0.0d || healthScore > 1.0d) {
            throw new IllegalArgumentException("healthScore must be between 0.0 and 1.0");
        }
        if (issueCount != issueTypes.size()) {
            throw new IllegalArgumentException("issueCount must match issueTypes size");
        }
        if (issueDetailCount != issueDetails.size()) {
            throw new IllegalArgumentException("issueDetailCount must match issueDetails size");
        }
        int computedBlockingIssueCount = (int) issueDetails.stream()
                .filter(detail -> Boolean.TRUE.equals(detail.get("blocking")))
                .count();
        if (blockingIssueCount != computedBlockingIssueCount) {
            throw new IllegalArgumentException("blockingIssueCount must match blocking issueDetails entries");
        }
        if (failingCheckCount != failingChecks.size()) {
            throw new IllegalArgumentException("failingCheckCount must match failingChecks size");
        }
        if (passingCheckCount + failingCheckCount != checks.size()) {
            throw new IllegalArgumentException("passingCheckCount + failingCheckCount must match checks size");
        }
        if (healthy && (!issueDetails.isEmpty() || !failingChecks.isEmpty())) {
            throw new IllegalArgumentException("healthy lineage health snapshots must not carry failures");
        }
        if (!healthy && primaryIssue.isEmpty()) {
            throw new IllegalArgumentException("unhealthy lineage health snapshots must carry a primaryIssue");
        }
        if (!failingChecks.isEmpty() && primaryFailingCheck.isEmpty()) {
            throw new IllegalArgumentException("snapshots with failing checks must carry a primaryFailingCheck");
        }
        if (!status.equals(String.valueOf(healthBadge.get("status")))) {
            throw new IllegalArgumentException("healthBadge status must match status");
        }
        if (!status.equals(String.valueOf(checkSummary.get("status")))) {
            throw new IllegalArgumentException("checkSummary status must match status");
        }
    }

    private static List<String> immutableStringList(List<?> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof CharSequence text) {
                        return text.toString();
                    }
                    throw new IllegalArgumentException(name + " entries must be strings");
                })
                .toList();
    }

    private static Optional<String> optionalString(Map<String, Object> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            return Optional.empty();
        }
        String value = String.valueOf(metadata.get(key));
        return value.isBlank() ? Optional.empty() : Optional.of(value);
    }

}
