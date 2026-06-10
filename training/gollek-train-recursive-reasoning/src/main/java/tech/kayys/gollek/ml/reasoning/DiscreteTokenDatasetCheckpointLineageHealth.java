package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

final class DiscreteTokenDatasetCheckpointLineageHealth {

    private static final String PAYLOAD_KIND = DiscreteTokenDatasetCheckpointLineageHealthSnapshot.KIND;
    private static final String SCHEMA_VERSION = DiscreteTokenDatasetCheckpointLineageHealthSnapshot.SCHEMA_VERSION;
    private static final String HEALTHY_CODE = "GOLLEK_LINEAGE_HEALTHY";
    private static final String NO_LINEAGE_ACTION = "No lineage action required.";
    private static final String NO_CHECK_ACTION = "No action required.";
    private static final String STATUS_HEALTHY = "healthy";
    private static final String STATUS_UNHEALTHY = "unhealthy";
    private static final String ALERT_INFO = "info";
    private static final String ALERT_CRITICAL = "critical";
    private static final String ALERT_WARNING = "warning";
    private static final String ISSUE_SEVERITY_ERROR = "error";
    private static final String CHECK_STATUS_PASS = "pass";
    private static final String CHECK_STATUS_FAIL = "fail";
    private static final String BADGE_VARIANT_SUCCESS = "success";
    private static final String BADGE_VARIANT_DANGER = "danger";
    private static final String BADGE_LABEL_HEALTHY = "Healthy";
    private static final String BADGE_LABEL_UNHEALTHY = "Unhealthy";
    private static final String BADGE_TOKEN_HEALTHY = "lineage-health-healthy";
    private static final String BADGE_TOKEN_UNHEALTHY = "lineage-health-unhealthy";
    private static final String NO_FAILING_CHECK_NAME = "none";
    private static final String CHECKS_PASSED_MESSAGE = "Lineage health checks passed.";
    private static final IssueDefinition DUPLICATE_RUN_ID = new IssueDefinition(
            "duplicate-run-id",
            "GOLLEK_LINEAGE_DUPLICATE_RUN_ID",
            10,
            "Keep one checkpoint for this run id or rewrite one manifest with a unique run id before resuming.");
    private static final IssueDefinition CYCLE = new IssueDefinition(
            "cycle",
            "GOLLEK_LINEAGE_CYCLE",
            20,
            "Fix checkpoint lineage parentRunId values so ancestry reaches a root without loops.");
    private static final IssueDefinition MISSING_PARENT = new IssueDefinition(
            "missing-parent",
            "GOLLEK_LINEAGE_MISSING_PARENT",
            30,
            "Restore the parent checkpoint into the same root or select a checkpoint whose ancestry is complete.");
    private static final IssueDefinition AMBIGUOUS_PARENT = new IssueDefinition(
            "ambiguous-parent",
            "GOLLEK_LINEAGE_AMBIGUOUS_PARENT",
            40,
            "Remove duplicate parent run ids or move unrelated checkpoints out of the checkpoint root.");
    private static final IssueDefinition PARENT_MISMATCH = new IssueDefinition(
            "parent-mismatch",
            "GOLLEK_LINEAGE_PARENT_MISMATCH",
            50,
            "Regenerate the child lineage from the resolved parent snapshot or resume from the matching parent checkpoint.");

    private DiscreteTokenDatasetCheckpointLineageHealth() {}

    static Map<String, Object> healthMetadata(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        List<Map<String, Object>> issueDetails = issueDetails(graph);
        Optional<Map<String, Object>> primaryIssue = primaryIssue(issueDetails);
        List<Map<String, Object>> checks = checks(graph);
        List<Map<String, Object>> failingChecks = failingChecks(checks);
        Optional<Map<String, Object>> primaryFailingCheck = primaryFailingCheck(failingChecks);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("kind", PAYLOAD_KIND);
        metadata.put("schemaVersion", SCHEMA_VERSION);
        metadata.put("rootDir", graph.rootDir().toString());
        metadata.put("status", status(graph));
        metadata.put("healthScore", healthScore(graph));
        metadata.put("alertLevel", alertLevel(graph));
        metadata.put("healthy", graph.healthy());
        metadata.put("issueCount", issueCount(graph));
        metadata.put("issueTypes", issueTypes(graph));
        metadata.put("issueDetailCount", issueDetails.size());
        metadata.put("issueCodes", issueCodes(issueDetails));
        metadata.put("blockingIssueCount", blockingIssueCount(issueDetails));
        primaryIssue.ifPresent(issue -> {
            metadata.put("primaryIssueCode", issue.get("code"));
            metadata.put("primaryIssueType", issue.get("type"));
            metadata.put("primaryIssueAction", issue.get("action"));
            metadata.put("primaryIssue", issue);
        });
        metadata.put("issueDetails", issueDetails);
        metadata.put("duplicateRunIds", graph.duplicateRunIds());
        metadata.put("missingParentRunIds", graph.missingParentRunIds());
        metadata.put("ambiguousParentRunIds", graph.ambiguousParentRunIds());
        metadata.put("parentMismatchRunIds", graph.parentMismatchRunIds());
        metadata.put("cycleRunIds", graph.cycleRunIds());
        metadata.put("unresolvedRunIds", graph.unresolvedNodes().stream()
                .map(DiscreteTokenDatasetCheckpointLineageGraph.Node::runId)
                .distinct()
                .sorted()
                .toList());
        metadata.put("summaryMessage", summaryMessage(graph));
        metadata.put("recommendedAction", recommendedAction(graph));
        metadata.put("healthBadge", healthBadge(graph));
        metadata.put("checks", checks);
        metadata.put("failingChecks", failingChecks);
        primaryFailingCheck.ifPresent(check -> {
            metadata.put("primaryFailingCheck", check);
            metadata.put("primaryFailingCheckName", check.get("name"));
            metadata.put("primaryFailingCheckType", check.get("type"));
            metadata.put("primaryFailingCheckCode", check.get("code"));
            metadata.put("primaryFailingCheckSeverity", check.get("severity"));
            metadata.put("primaryFailingCheckAction", check.get("action"));
            metadata.put("primaryFailingCheckMessage", check.get("detail"));
        });
        metadata.put("passingCheckCount", passingCheckCount(checks));
        metadata.put("failingCheckCount", failingChecks.size());
        metadata.put("checkSummary", checkSummary(graph, checks, failingChecks));
        metadata.put("summary", graph.healthSummary());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    static String status(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return requireGraph(graph).healthy() ? STATUS_HEALTHY : STATUS_UNHEALTHY;
    }

    static double healthScore(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return requireGraph(graph).healthy() ? 1.0d : 0.0d;
    }

    static String alertLevel(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return requireGraph(graph).healthy() ? ALERT_INFO : ALERT_CRITICAL;
    }

    static String summaryMessage(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return requireGraph(graph).healthSummary();
    }

    static String recommendedAction(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        return primaryIssueAction(graph).orElse(NO_LINEAGE_ACTION);
    }

    static Map<String, Object> healthBadge(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        List<Map<String, Object>> failingChecks = failingChecks(graph);
        Map<String, Object> badge = new LinkedHashMap<>();
        badge.put("status", status(graph));
        badge.put("alertLevel", alertLevel(graph));
        badge.put("primaryIssueCode", primaryIssueCode(graph).orElse(HEALTHY_CODE));
        badge.put("primaryCheckCode", primaryFailingCheckCode(failingChecks));
        badge.put("primaryCheckType", primaryFailingCheckType(failingChecks));
        badge.put("primaryCheckSeverity", primaryFailingCheckSeverity(failingChecks));
        badge.put("primaryCheckAction", primaryFailingCheckAction(failingChecks));
        badge.put("score", healthScore(graph));
        badge.put("variant", graph.healthy() ? BADGE_VARIANT_SUCCESS : BADGE_VARIANT_DANGER);
        badge.put("label", graph.healthy() ? BADGE_LABEL_HEALTHY : BADGE_LABEL_UNHEALTHY);
        badge.put("token", graph.healthy() ? BADGE_TOKEN_HEALTHY : BADGE_TOKEN_UNHEALTHY);
        badge.put("tooltip", summaryMessage(graph));
        badge.put("checkStatus", graph.healthy() ? CHECK_STATUS_PASS : CHECK_STATUS_FAIL);
        badge.put("recommendedAction", recommendedAction(graph));
        return Collections.unmodifiableMap(new LinkedHashMap<>(badge));
    }

    static List<Map<String, Object>> checks(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        List<Map<String, Object>> checks = new ArrayList<>();
        checks.add(check(
                DUPLICATE_RUN_ID,
                "uniqueRunIds",
                "Unique run ids",
                graph.duplicateRunIds(),
                graph.duplicateRunIds(),
                "Every checkpoint run id is unique.",
                "Duplicate checkpoint run ids were found."));
        checks.add(check(
                MISSING_PARENT,
                "parentExists",
                "Parent exists",
                graph.missingParentRunIds(),
                graph.nodes().stream()
                        .filter(node -> node.parentRunId() != null && !node.parentPresent())
                        .map(DiscreteTokenDatasetCheckpointLineageGraph.Node::runId)
                        .distinct()
                        .sorted()
                        .toList(),
                "Every non-root checkpoint resolves to an existing parent run id.",
                "One or more non-root checkpoints reference a missing parent run id."));
        checks.add(check(
                AMBIGUOUS_PARENT,
                "parentUnique",
                "Parent unique",
                graph.ambiguousParentRunIds(),
                graph.nodes().stream()
                        .filter(DiscreteTokenDatasetCheckpointLineageGraph.Node::parentAmbiguous)
                        .map(DiscreteTokenDatasetCheckpointLineageGraph.Node::runId)
                        .distinct()
                        .sorted()
                        .toList(),
                "Every parent run id resolves to exactly one checkpoint.",
                "One or more parent run ids resolve to multiple checkpoints."));
        checks.add(check(
                PARENT_MISMATCH,
                "parentIdentity",
                "Parent identity",
                graph.parentMismatchRunIds(),
                graph.parentMismatchRunIds(),
                "Recorded parent step and dataset fingerprint match resolved parents.",
                "Recorded parent identity does not match one or more resolved parent checkpoints."));
        checks.add(check(
                CYCLE,
                "acyclic",
                "Acyclic lineage",
                graph.cycleRunIds(),
                graph.cycleRunIds(),
                "Checkpoint ancestry reaches a root without loops.",
                "One or more checkpoints participate in a lineage cycle."));
        return List.copyOf(checks);
    }

    static List<Map<String, Object>> failingChecks(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return failingChecks(checks(graph));
    }

    static int passingCheckCount(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return passingCheckCount(checks(graph));
    }

    static int failingCheckCount(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return failingChecks(graph).size();
    }

    static Map<String, Object> checkSummary(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        List<Map<String, Object>> checks = checks(graph);
        return checkSummary(graph, checks, failingChecks(checks));
    }

    static Optional<Map<String, Object>> primaryFailingCheck(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheck(failingChecks(graph));
    }

    static Optional<String> primaryFailingCheckName(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "name");
    }

    static Optional<String> primaryFailingCheckType(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "type");
    }

    static Optional<String> primaryFailingCheckCode(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "code");
    }

    static Optional<String> primaryFailingCheckSeverity(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "severity");
    }

    static Optional<String> primaryFailingCheckAction(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "action");
    }

    static Optional<String> primaryFailingCheckMessage(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryFailingCheckField(graph, "detail");
    }

    static int issueCount(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        int count = 0;
        if (!graph.duplicateRunIds().isEmpty()) {
            count++;
        }
        if (!graph.missingParentRunIds().isEmpty()) {
            count++;
        }
        if (!graph.ambiguousParentRunIds().isEmpty()) {
            count++;
        }
        if (!graph.parentMismatchRunIds().isEmpty()) {
            count++;
        }
        if (!graph.cycleRunIds().isEmpty()) {
            count++;
        }
        return count;
    }

    static List<String> issueTypes(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        List<String> types = new ArrayList<>();
        if (!graph.duplicateRunIds().isEmpty()) {
            types.add(DUPLICATE_RUN_ID.type());
        }
        if (!graph.missingParentRunIds().isEmpty()) {
            types.add(MISSING_PARENT.type());
        }
        if (!graph.ambiguousParentRunIds().isEmpty()) {
            types.add(AMBIGUOUS_PARENT.type());
        }
        if (!graph.parentMismatchRunIds().isEmpty()) {
            types.add(PARENT_MISMATCH.type());
        }
        if (!graph.cycleRunIds().isEmpty()) {
            types.add(CYCLE.type());
        }
        return List.copyOf(types);
    }

    static int issueDetailCount(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return issueDetails(graph).size();
    }

    static List<String> issueCodes(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return issueCodes(issueDetails(graph));
    }

    static int blockingIssueCount(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return blockingIssueCount(issueDetails(graph));
    }

    static Optional<Map<String, Object>> primaryIssue(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryIssue(issueDetails(graph));
    }

    static Optional<String> primaryIssueCode(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryIssue(graph).map(issue -> String.valueOf(issue.get("code")));
    }

    static Optional<String> primaryIssueType(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryIssue(graph).map(issue -> String.valueOf(issue.get("type")));
    }

    static Optional<String> primaryIssueAction(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return primaryIssue(graph).map(issue -> String.valueOf(issue.get("action")));
    }

    static List<Map<String, Object>> issueDetails(DiscreteTokenDatasetCheckpointLineageGraph graph) {
        graph = requireGraph(graph);
        if (graph.healthy()) {
            return List.of();
        }
        Map<String, List<DiscreteTokenDatasetCheckpointLineageGraph.Node>> nodesByRunId = nodesByRunId(graph);
        List<Map<String, Object>> details = new ArrayList<>();
        for (String runId : graph.duplicateRunIds()) {
            List<DiscreteTokenDatasetCheckpointLineageGraph.Node> matches = nodesByRunId.getOrDefault(runId, List.of());
            Map<String, Object> issue = new LinkedHashMap<>();
            issue.put("type", DUPLICATE_RUN_ID.type());
            issue.put("code", DUPLICATE_RUN_ID.code());
            issue.put("severity", ISSUE_SEVERITY_ERROR);
            issue.put("blocking", true);
            issue.put("runId", runId);
            issue.put("checkpointDirs", matches.stream()
                    .map(node -> node.checkpointDir().toString())
                    .sorted()
                    .toList());
            issue.put("detail", "run id appears in " + matches.size() + " checkpoints");
            issue.put("action", DUPLICATE_RUN_ID.action());
            details.add(Collections.unmodifiableMap(new LinkedHashMap<>(issue)));
        }
        for (DiscreteTokenDatasetCheckpointLineageGraph.Node node : graph.nodes()) {
            if (node.parentRunId() != null && !node.parentPresent()) {
                Map<String, Object> issue = issueDetail(MISSING_PARENT, node);
                issue.put("parentRunId", node.parentRunId());
                issue.put("detail", "parent run id was not found under checkpoint root");
                details.add(Collections.unmodifiableMap(new LinkedHashMap<>(issue)));
            }
            if (node.parentAmbiguous()) {
                List<DiscreteTokenDatasetCheckpointLineageGraph.Node> parents =
                        nodesByRunId.getOrDefault(node.parentRunId(), List.of());
                Map<String, Object> issue = issueDetail(AMBIGUOUS_PARENT, node);
                issue.put("parentRunId", node.parentRunId());
                issue.put("parentCheckpointDirs", parents.stream()
                        .map(parent -> parent.checkpointDir().toString())
                        .sorted()
                        .toList());
                issue.put("detail", "parent run id resolves to " + parents.size() + " checkpoints");
                details.add(Collections.unmodifiableMap(new LinkedHashMap<>(issue)));
            }
            if (!node.parentIdentityMatched()) {
                Map<String, Object> issue = issueDetail(PARENT_MISMATCH, node);
                issue.put("parentRunId", node.parentRunId());
                issue.put("reasons", node.parentMismatchReasons());
                issue.put("detail", "recorded parent identity does not match resolved parent checkpoint");
                details.add(Collections.unmodifiableMap(new LinkedHashMap<>(issue)));
            }
            if (node.cycleMember()) {
                Map<String, Object> issue = issueDetail(CYCLE, node);
                issue.put("parentRunId", node.parentRunId());
                issue.put("detail", "node participates in a lineage cycle");
                details.add(Collections.unmodifiableMap(new LinkedHashMap<>(issue)));
            }
        }
        return List.copyOf(details);
    }

    private static List<String> issueCodes(List<Map<String, Object>> issueDetails) {
        return issueDetails.stream()
                .map(detail -> String.valueOf(detail.get("code")))
                .distinct()
                .toList();
    }

    private static int blockingIssueCount(List<Map<String, Object>> issueDetails) {
        return (int) issueDetails.stream()
                .filter(detail -> Boolean.TRUE.equals(detail.get("blocking")))
                .count();
    }

    private static List<Map<String, Object>> failingChecks(List<Map<String, Object>> checks) {
        return checks.stream()
                .filter(check -> !Boolean.TRUE.equals(check.get("passed")))
                .toList();
    }

    private static int passingCheckCount(List<Map<String, Object>> checks) {
        return (int) checks.stream()
                .filter(check -> Boolean.TRUE.equals(check.get("passed")))
                .count();
    }

    private static Map<String, Object> checkSummary(
            DiscreteTokenDatasetCheckpointLineageGraph graph,
            List<Map<String, Object>> checks,
            List<Map<String, Object>> failingChecks) {
        int passingCheckCount = passingCheckCount(checks);
        Optional<Map<String, Object>> primaryFailingCheck = primaryFailingCheck(failingChecks);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("total", checks.size());
        summary.put("passed", passingCheckCount);
        summary.put("failed", failingChecks.size());
        summary.put("passRate", checks.isEmpty() ? 1.0d : (double) passingCheckCount / (double) checks.size());
        summary.put("failureRate", checks.isEmpty() ? 0.0d : (double) failingChecks.size() / (double) checks.size());
        summary.put("status", status(graph));
        summary.put("alertLevel", alertLevel(graph));
        summary.put("summaryMessage", failingChecks.isEmpty()
                ? CHECKS_PASSED_MESSAGE
                : "Lineage health checks reported " + failingChecks.size() + " failure(s).");
        summary.put("recommendedAction", recommendedAction(graph));
        summary.put("issueCodes", failingChecks.stream()
                .map(check -> String.valueOf(check.get("code")))
                .toList());
        summary.put("primaryIssueCode", primaryIssueCode(graph).orElse(HEALTHY_CODE));
        summary.put("allPassed", failingChecks.isEmpty());
        summary.put("hasCriticalFailures", !failingChecks.isEmpty());
        summary.put("hasWarningFailures", hasChecksBySeverity(failingChecks, ALERT_WARNING));
        summary.put("hasInfoFailures", hasChecksBySeverity(failingChecks, ALERT_INFO));
        summary.put("dominantSeverity", dominantSeverity(failingChecks));
        summary.put("failingSeverityCounts", severityCounts(failingChecks));
        summary.put("severityCounts", severityCounts(checks));
        summary.put("primaryFailingName", primaryFailingCheckValue(failingChecks, "name", NO_FAILING_CHECK_NAME));
        summary.put("primaryFailingType", primaryFailingCheckType(failingChecks));
        summary.put("primaryFailingCode", primaryFailingCheckCode(failingChecks));
        summary.put("primaryFailingSeverity", primaryFailingCheckSeverity(failingChecks));
        summary.put("primaryFailingAction", primaryFailingCheckAction(failingChecks));
        summary.put("primaryFailingMessage", primaryFailingCheckValue(
                failingChecks,
                "detail",
                CHECKS_PASSED_MESSAGE));
        primaryFailingCheck.ifPresent(check -> summary.put("primaryFailingCheck", check));
        summary.put("failingNames", failingChecks.stream()
                .map(check -> String.valueOf(check.get("name")))
                .toList());
        summary.put("failingCodes", failingChecks.stream()
                .map(check -> String.valueOf(check.get("code")))
                .toList());
        return Collections.unmodifiableMap(new LinkedHashMap<>(summary));
    }

    private static Optional<Map<String, Object>> primaryIssue(List<Map<String, Object>> issueDetails) {
        return issueDetails.stream()
                .min(Comparator
                        .comparingInt((Map<String, Object> detail) -> issuePriority(String.valueOf(detail.get("type"))))
                        .thenComparing(detail -> String.valueOf(detail.getOrDefault("code", "")))
                        .thenComparing(detail -> String.valueOf(detail.getOrDefault("runId", "")))
                        .thenComparing(detail -> String.valueOf(detail.getOrDefault("checkpointDir", ""))));
    }

    private static int issuePriority(String type) {
        return issueDefinition(type)
                .map(IssueDefinition::priority)
                .orElse(100);
    }

    private static Map<String, Object> issueDetail(
            IssueDefinition issue,
            DiscreteTokenDatasetCheckpointLineageGraph.Node node) {
        Map<String, Object> detail = new LinkedHashMap<>();
        detail.put("type", issue.type());
        detail.put("code", issue.code());
        detail.put("severity", ISSUE_SEVERITY_ERROR);
        detail.put("blocking", true);
        detail.put("runId", node.runId());
        detail.put("checkpointDir", node.checkpointDir().toString());
        detail.put("action", issue.action());
        return detail;
    }

    private static Map<String, Object> check(
            IssueDefinition issue,
            String name,
            String label,
            List<String> runIds,
            List<String> affectedRunIds,
            String passDetail,
            String failDetail) {
        boolean passed = runIds.isEmpty();
        Map<String, Object> check = new LinkedHashMap<>();
        check.put("name", name);
        check.put("type", issue.type());
        check.put("code", issue.code());
        check.put("label", label);
        check.put("passed", passed);
        check.put("status", passed ? CHECK_STATUS_PASS : CHECK_STATUS_FAIL);
        check.put("severity", passed ? ALERT_INFO : ALERT_CRITICAL);
        check.put("blocking", !passed);
        check.put("count", runIds.size());
        check.put("runIds", runIds);
        check.put("affectedRunIds", affectedRunIds);
        check.put("detail", passed ? passDetail : failDetail);
        check.put("action", passed ? NO_CHECK_ACTION : issue.action());
        return Collections.unmodifiableMap(new LinkedHashMap<>(check));
    }

    private static Map<String, Object> severityCounts(List<Map<String, Object>> checks) {
        Map<String, Object> counts = new LinkedHashMap<>();
        counts.put("critical", countChecksBySeverity(checks, ALERT_CRITICAL));
        counts.put("warning", countChecksBySeverity(checks, ALERT_WARNING));
        counts.put("info", countChecksBySeverity(checks, ALERT_INFO));
        return Collections.unmodifiableMap(new LinkedHashMap<>(counts));
    }

    private static long countChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return checks.stream()
                .filter(check -> severity.equals(String.valueOf(check.get("severity"))))
                .count();
    }

    private static boolean hasChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return countChecksBySeverity(checks, severity) > 0L;
    }

    private static String dominantSeverity(List<Map<String, Object>> failingChecks) {
        if (hasChecksBySeverity(failingChecks, ALERT_CRITICAL)) {
            return ALERT_CRITICAL;
        }
        if (hasChecksBySeverity(failingChecks, ALERT_WARNING)) {
            return ALERT_WARNING;
        }
        if (hasChecksBySeverity(failingChecks, ALERT_INFO)) {
            return ALERT_INFO;
        }
        return ALERT_INFO;
    }

    private static String primaryFailingCheckCode(List<Map<String, Object>> failingChecks) {
        return primaryFailingCheckValue(failingChecks, "code", HEALTHY_CODE);
    }

    private static String primaryFailingCheckType(List<Map<String, Object>> failingChecks) {
        return primaryFailingCheckValue(failingChecks, "type", NO_FAILING_CHECK_NAME);
    }

    private static String primaryFailingCheckSeverity(List<Map<String, Object>> failingChecks) {
        return primaryFailingCheckValue(failingChecks, "severity", ALERT_INFO);
    }

    private static String primaryFailingCheckAction(List<Map<String, Object>> failingChecks) {
        return primaryFailingCheckValue(failingChecks, "action", NO_CHECK_ACTION);
    }

    private static String primaryFailingCheckValue(
            List<Map<String, Object>> failingChecks,
            String key,
            String defaultValue) {
        return primaryFailingCheck(failingChecks)
                .map(check -> String.valueOf(check.getOrDefault(key, "")))
                .filter(value -> !value.isBlank())
                .orElse(defaultValue);
    }

    private static Optional<Map<String, Object>> primaryFailingCheck(List<Map<String, Object>> failingChecks) {
        return failingChecks.stream()
                .min(Comparator
                        .comparingInt((Map<String, Object> check) -> issuePriority(String.valueOf(check.get("type"))))
                        .thenComparing(check -> String.valueOf(check.getOrDefault("code", "")))
                        .thenComparing(check -> String.valueOf(check.getOrDefault("name", ""))));
    }

    private static Optional<String> primaryFailingCheckField(
            DiscreteTokenDatasetCheckpointLineageGraph graph,
            String key) {
        return primaryFailingCheck(graph)
                .map(check -> String.valueOf(check.getOrDefault(key, "")))
                .filter(value -> !value.isBlank());
    }

    private static Optional<IssueDefinition> issueDefinition(String type) {
        return switch (type) {
            case "duplicate-run-id" -> Optional.of(DUPLICATE_RUN_ID);
            case "missing-parent" -> Optional.of(MISSING_PARENT);
            case "ambiguous-parent" -> Optional.of(AMBIGUOUS_PARENT);
            case "parent-mismatch" -> Optional.of(PARENT_MISMATCH);
            case "cycle" -> Optional.of(CYCLE);
            default -> Optional.empty();
        };
    }

    private static Map<String, List<DiscreteTokenDatasetCheckpointLineageGraph.Node>> nodesByRunId(
            DiscreteTokenDatasetCheckpointLineageGraph graph) {
        Map<String, List<DiscreteTokenDatasetCheckpointLineageGraph.Node>> byRunId = new LinkedHashMap<>();
        for (DiscreteTokenDatasetCheckpointLineageGraph.Node node : graph.nodes()) {
            byRunId.computeIfAbsent(node.runId(), ignored -> new ArrayList<>()).add(node);
        }
        return Collections.unmodifiableMap(byRunId);
    }

    private static DiscreteTokenDatasetCheckpointLineageGraph requireGraph(
            DiscreteTokenDatasetCheckpointLineageGraph graph) {
        return Objects.requireNonNull(graph, "graph must not be null");
    }

    private record IssueDefinition(String type, String code, int priority, String action) {}
}
