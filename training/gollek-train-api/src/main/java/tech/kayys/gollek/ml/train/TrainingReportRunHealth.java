package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.mapValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Type-safe report view for trainer run health and promotion gate readiness.
 */
public record TrainingReportRunHealth(
        String status,
        boolean healthy,
        boolean gatePassed,
        boolean issueDetected,
        int issueCount,
        boolean blockingIssueDetected,
        int blockingIssueCount,
        String recommendedAction,
        Map<String, Object> primaryIssue,
        Map<String, Object> primaryBlockingIssue,
        List<String> issueCodes,
        List<String> issueSeverities,
        Map<String, Object> issueCountsByKind,
        Map<String, Object> issueCountsBySeverity,
        List<Map<String, Object>> issues) {
    public TrainingReportRunHealth {
        status = status == null || status.isBlank() ? "healthy" : status.trim();
        recommendedAction = recommendedAction == null || recommendedAction.isBlank()
                ? "continue monitoring training"
                : recommendedAction.trim();
        primaryIssue = snapshotMap(primaryIssue);
        primaryBlockingIssue = snapshotMap(primaryBlockingIssue);
        issueCodes = List.copyOf(issueCodes == null ? List.of() : issueCodes);
        issueSeverities = List.copyOf(issueSeverities == null ? List.of() : issueSeverities);
        issueCountsByKind = snapshotMap(issueCountsByKind);
        issueCountsBySeverity = snapshotMap(issueCountsBySeverity);
        issues = snapshotIssueList(issues);
    }

    public static TrainingReportRunHealth defaultHealthy() {
        return new TrainingReportRunHealth(
                "healthy",
                true,
                true,
                false,
                0,
                false,
                0,
                "continue monitoring training",
                Map.of(),
                Map.of(),
                List.of(),
                List.of(),
                Map.of(),
                Map.of(),
                List.of());
    }

    public static TrainingReportRunHealth fromMetadata(Map<String, ?> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return defaultHealthy();
        }
        boolean issueDetected = booleanValue(metadata.get("trainingRunIssueDetected"));
        int issueCount = intValue(metadata.get("trainingRunIssueCount"), 0);
        boolean blockingIssueDetected = booleanValue(metadata.get("trainingRunBlockingIssueDetected"));
        int blockingIssueCount = intValue(metadata.get("trainingRunBlockingIssueCount"), 0);
        return new TrainingReportRunHealth(
                stringValue(metadata.get("trainingRunHealthStatus"), issueDetected ? "warning" : "healthy"),
                metadata.containsKey("trainingRunHealthy")
                        ? booleanValue(metadata.get("trainingRunHealthy"))
                        : !issueDetected,
                !metadata.containsKey("trainingRunGatePassed")
                        || booleanValue(metadata.get("trainingRunGatePassed")),
                issueDetected,
                issueCount,
                blockingIssueDetected,
                blockingIssueCount,
                stringValue(metadata.get("trainingRunRecommendedAction"), "continue monitoring training"),
                primaryIssue(metadata, "trainingRunPrimaryIssue"),
                primaryIssue(metadata, "trainingRunPrimaryBlockingIssue"),
                stringList(metadata.get("trainingRunIssueCodes")),
                stringList(metadata.get("trainingRunIssueSeverities")),
                mapValue(metadata, "trainingRunIssueCountsByKind"),
                mapValue(metadata, "trainingRunIssueCountsBySeverity"),
                issueList(metadata.get("trainingRunIssues")));
    }

    public static TrainingReportRunHealth fromMap(Map<String, ?> runHealth) {
        if (runHealth == null || runHealth.isEmpty()) {
            return defaultHealthy();
        }
        return new TrainingReportRunHealth(
                stringValue(runHealth.get("status"), "healthy"),
                runHealth.containsKey("healthy")
                        ? booleanValue(runHealth.get("healthy"))
                        : !booleanValue(runHealth.get("issueDetected")),
                !runHealth.containsKey("gatePassed") || booleanValue(runHealth.get("gatePassed")),
                booleanValue(runHealth.get("issueDetected")),
                intValue(runHealth.get("issueCount"), 0),
                booleanValue(runHealth.get("blockingIssueDetected")),
                intValue(runHealth.get("blockingIssueCount"), 0),
                stringValue(runHealth.get("recommendedAction"), "continue monitoring training"),
                mapValue(runHealth, "primaryIssue"),
                mapValue(runHealth, "primaryBlockingIssue"),
                stringList(runHealth.get("issueCodes")),
                stringList(runHealth.get("issueSeverities")),
                mapValue(runHealth, "issueCountsByKind"),
                mapValue(runHealth, "issueCountsBySeverity"),
                issueList(runHealth.get("issues")));
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("status", status);
        map.put("healthy", healthy);
        map.put("gatePassed", gatePassed);
        map.put("issueDetected", issueDetected);
        map.put("issueCount", issueCount);
        map.put("blockingIssueDetected", blockingIssueDetected);
        map.put("blockingIssueCount", blockingIssueCount);
        map.put("recommendedAction", recommendedAction);
        map.put("primaryIssue", primaryIssue);
        map.put("primaryBlockingIssue", primaryBlockingIssue);
        map.put("issueCodes", issueCodes);
        map.put("issueSeverities", issueSeverities);
        map.put("issueCountsByKind", issueCountsByKind);
        map.put("issueCountsBySeverity", issueCountsBySeverity);
        map.put("issues", issues);
        return Map.copyOf(map);
    }

    private static Map<String, Object> primaryIssue(Map<String, ?> metadata, String prefix) {
        if (!booleanValue(metadata.get(prefix + "Available"))) {
            return Map.of();
        }
        Map<String, Object> issue = new LinkedHashMap<>();
        putIfPresent(issue, "kind", metadata.get(prefix + "Kind"));
        putIfPresent(issue, "code", metadata.get(prefix + "Code"));
        putIfPresent(issue, "severity", metadata.get(prefix + "Severity"));
        putIfPresent(issue, "blocking", metadata.get(prefix + "Blocking"));
        putIfPresent(issue, "artifact", metadata.get(prefix + "Artifact"));
        putIfPresent(issue, "message", metadata.get(prefix + "Message"));
        putIfPresent(issue, "action", metadata.get(prefix + "RecommendedAction"));
        return Map.copyOf(issue);
    }

    private static void putIfPresent(Map<String, Object> issue, String key, Object value) {
        if (value != null) {
            issue.put(key, value);
        }
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item != null) {
                values.add(String.valueOf(item));
            }
        }
        return List.copyOf(values);
    }

    private static List<Map<String, Object>> issueList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Map<String, Object>> issues = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                issues.add(snapshotMap(map));
            }
        }
        return List.copyOf(issues);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> snapshotMap(Map<?, ?> map) {
        if (map == null || map.isEmpty()) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        return snapshot instanceof Map<?, ?> snapshotMap
                ? (Map<String, Object>) snapshotMap
                : Map.of();
    }

    private static List<Map<String, Object>> snapshotIssueList(List<Map<String, Object>> issues) {
        if (issues == null || issues.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> snapshots = new ArrayList<>();
        for (Map<String, Object> issue : issues) {
            snapshots.add(snapshotMap(issue));
        }
        return List.copyOf(snapshots);
    }
}
