package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Type-safe view over one trainer health section such as data-loader or data-distribution health.
 */
public record TrainingReportHealthStatus(
        boolean available,
        String skipReason,
        String status,
        boolean healthy,
        boolean gatePassed,
        boolean issueDetected,
        int issueCount,
        int warningCount,
        int errorCount,
        List<String> issueCodes,
        List<String> issueSeverities,
        List<String> recommendedActions,
        List<Map<String, Object>> issues) {
    public TrainingReportHealthStatus {
        skipReason = skipReason == null ? "" : skipReason.trim();
        status = status == null || status.isBlank() ? "unknown" : status.trim();
        issueCodes = List.copyOf(issueCodes == null ? List.of() : issueCodes);
        issueSeverities = List.copyOf(issueSeverities == null ? List.of() : issueSeverities);
        recommendedActions = List.copyOf(recommendedActions == null ? List.of() : recommendedActions);
        issues = snapshotIssueList(issues);
    }

    public static TrainingReportHealthStatus fromMetadata(Map<String, ?> metadata, String prefix) {
        return fromMetadata(metadata, prefix, prefix + "-metadata-missing");
    }

    public static TrainingReportHealthStatus fromMetadata(Map<String, ?> metadata, String prefix, String missingReason) {
        if (metadata == null || metadata.isEmpty() || !metadata.containsKey(prefix + ".available")) {
            return unknown(missingReason);
        }
        boolean issueDetected = booleanValue(metadata.get(prefix + "IssueDetected"));
        int issueCount = intValue(metadata.get(prefix + "IssueCount"), 0);
        boolean available = booleanValue(metadata.get(prefix + ".available"));
        String status = stringValue(metadata.get(prefix + "Status"), available ? "healthy" : "unknown");
        return new TrainingReportHealthStatus(
                available,
                stringValue(metadata.get(prefix + ".skipReason"), ""),
                status,
                metadata.containsKey(prefix + "Healthy")
                        ? booleanValue(metadata.get(prefix + "Healthy"))
                        : "healthy".equals(status) && issueCount == 0,
                !metadata.containsKey(prefix + "GatePassed") || booleanValue(metadata.get(prefix + "GatePassed")),
                issueDetected,
                issueCount,
                intValue(metadata.get(prefix + "WarningCount"), 0),
                intValue(metadata.get(prefix + "ErrorCount"), 0),
                stringList(metadata.get(prefix + "IssueCodes")),
                stringList(metadata.get(prefix + "IssueSeverities")),
                stringList(metadata.get(prefix + "RecommendedActions")),
                issueList(metadata.get(prefix + "Issues")));
    }

    public static TrainingReportHealthStatus fromMap(Map<String, ?> value) {
        return fromMap(value, "health-metadata-missing");
    }

    public static TrainingReportHealthStatus fromMap(Map<String, ?> value, String missingReason) {
        if (value == null || value.isEmpty()) {
            return unknown(missingReason);
        }
        boolean issueDetected = booleanValue(value.get("issueDetected"));
        int issueCount = intValue(value.get("issueCount"), 0);
        String status = stringValue(value.get("status"), "unknown");
        return new TrainingReportHealthStatus(
                booleanValue(value.get("available")),
                stringValue(value.get("skipReason"), ""),
                status,
                value.containsKey("healthy")
                        ? booleanValue(value.get("healthy"))
                        : "healthy".equals(status) && issueCount == 0,
                !value.containsKey("gatePassed") || booleanValue(value.get("gatePassed")),
                issueDetected,
                issueCount,
                intValue(value.get("warningCount"), 0),
                intValue(value.get("errorCount"), 0),
                stringList(value.get("issueCodes")),
                stringList(value.get("issueSeverities")),
                stringList(value.get("recommendedActions")),
                issueList(value.get("issues")));
    }

    public static TrainingReportHealthStatus unknown(String reason) {
        return new TrainingReportHealthStatus(
                false,
                reason,
                "unknown",
                false,
                true,
                false,
                0,
                0,
                0,
                List.of(),
                List.of(),
                List.of(),
                List.of());
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available);
        map.put("skipReason", skipReason);
        map.put("status", status);
        map.put("healthy", healthy);
        map.put("gatePassed", gatePassed);
        map.put("issueDetected", issueDetected);
        map.put("issueCount", issueCount);
        map.put("warningCount", warningCount);
        map.put("errorCount", errorCount);
        map.put("issueCodes", issueCodes);
        map.put("issueSeverities", issueSeverities);
        map.put("recommendedActions", recommendedActions);
        map.put("issues", issues);
        return Map.copyOf(map);
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
