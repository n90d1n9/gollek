package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Type-safe report view for trainer data-loader and distribution health.
 */
public record TrainingReportDataHealth(
        TrainingReportHealthStatus loaderPlan,
        TrainingReportHealthStatus distribution) {
    public TrainingReportDataHealth {
        loaderPlan = loaderPlan == null
                ? TrainingReportHealthStatus.unknown("data-loader-plan-health-metadata-missing")
                : loaderPlan;
        distribution = distribution == null
                ? TrainingReportHealthStatus.unknown("data-distribution-health-metadata-missing")
                : distribution;
    }

    public static TrainingReportDataHealth fromMetadata(Map<String, ?> metadata) {
        return new TrainingReportDataHealth(
                TrainingReportHealthStatus.fromMetadata(
                        metadata,
                        "dataLoaderPlanHealth",
                        "data-loader-plan-health-metadata-missing"),
                TrainingReportHealthStatus.fromMetadata(
                        metadata,
                        "dataDistributionHealth",
                        "data-distribution-health-metadata-missing"));
    }

    public static TrainingReportDataHealth fromMap(Map<String, ?> dataHealth) {
        if (dataHealth == null || dataHealth.isEmpty()) {
            return fromMetadata(Map.of());
        }
        return new TrainingReportDataHealth(
                TrainingReportHealthStatus.fromMap(
                        mapValue(dataHealth.get("loaderPlan")),
                        "data-loader-plan-health-metadata-missing"),
                TrainingReportHealthStatus.fromMap(
                        mapValue(dataHealth.get("distribution")),
                        "data-distribution-health-metadata-missing"));
    }

    public boolean gatePassed() {
        return loaderPlan.gatePassed() && distribution.gatePassed();
    }

    public boolean available() {
        return loaderPlan.available() || distribution.available();
    }

    public boolean healthy() {
        return loaderPlan.healthy() && distribution.healthy();
    }

    public boolean issueDetected() {
        return loaderPlan.issueDetected() || distribution.issueDetected();
    }

    public int issueCount() {
        return loaderPlan.issueCount() + distribution.issueCount();
    }

    public int warningCount() {
        return loaderPlan.warningCount() + distribution.warningCount();
    }

    public int errorCount() {
        return loaderPlan.errorCount() + distribution.errorCount();
    }

    public List<String> issueCodes() {
        return combine(loaderPlan.issueCodes(), distribution.issueCodes());
    }

    public List<String> issueSeverities() {
        return combine(loaderPlan.issueSeverities(), distribution.issueSeverities());
    }

    public List<String> recommendedActions() {
        return combine(loaderPlan.recommendedActions(), distribution.recommendedActions());
    }

    public String issueSummary(String fallback) {
        String severity = firstNonBlank(issueSeverities());
        String code = firstNonBlank(issueCodes());
        String action = firstNonBlank(recommendedActions());
        if (!severity.isBlank()) {
            return issueSummaryWithSeverity(severity, code, action);
        }
        if (errorCount() > 0) {
            return issueSummaryWithSeverity("error", code, action);
        }
        if (warningCount() > 0) {
            return issueSummaryWithSeverity("warning", code, action);
        }
        if (!code.isBlank()) {
            return "inspect data-health issue " + code;
        }
        if (!action.isBlank()) {
            return action;
        }
        return fallback == null || fallback.isBlank()
                ? "inspect trainer data-loader and distribution health"
                : fallback.trim();
    }

    public List<Map<String, Object>> issues() {
        List<Map<String, Object>> issues = new ArrayList<>();
        issues.addAll(loaderPlan.issues());
        issues.addAll(distribution.issues());
        return List.copyOf(issues);
    }

    public List<Map<String, Object>> issueSummaries() {
        return TrainingReportHealthIssueSummaries.from(issues());
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("loaderPlan", loaderPlan.toMap());
        map.put("distribution", distribution.toMap());
        map.put("available", available());
        map.put("healthy", healthy());
        map.put("gatePassed", gatePassed());
        map.put("issueDetected", issueDetected());
        map.put("issueCount", issueCount());
        map.put("warningCount", warningCount());
        map.put("errorCount", errorCount());
        map.put("issueCodes", issueCodes());
        map.put("issueSeverities", issueSeverities());
        map.put("recommendedActions", recommendedActions());
        map.put("issues", issues());
        map.put("issueSummaries", issueSummaries());
        return Map.copyOf(map);
    }

    private static List<String> combine(List<String> left, List<String> right) {
        List<String> values = new ArrayList<>();
        values.addAll(left == null ? List.of() : left);
        values.addAll(right == null ? List.of() : right);
        return List.copyOf(values);
    }

    private static String issueSummaryWithSeverity(String severity, String code, String action) {
        String normalizedSeverity = severity.trim().toLowerCase(Locale.ROOT);
        if (!code.isBlank() && !action.isBlank()) {
            return normalizedSeverity + " " + code + " - " + action;
        }
        if (!code.isBlank()) {
            return normalizedSeverity + " " + code;
        }
        if (!action.isBlank()) {
            return normalizedSeverity + " - " + action;
        }
        return normalizedSeverity;
    }

    private static String firstNonBlank(List<String> values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return "";
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapValue(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        return snapshot instanceof Map<?, ?> snapshotMap
                ? (Map<String, Object>) snapshotMap
                : Map.of();
    }
}
