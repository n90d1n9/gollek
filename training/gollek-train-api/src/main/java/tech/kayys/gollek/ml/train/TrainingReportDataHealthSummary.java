package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Compact, export-friendly data-health summary shared by validation and portfolio views.
 */
public record TrainingReportDataHealthSummary(
        boolean available,
        boolean gatePassed,
        boolean healthy,
        boolean issueDetected,
        int issueCount,
        int warningCount,
        int errorCount,
        String status,
        List<String> issueCodes) {
    public TrainingReportDataHealthSummary {
        status = status == null || status.isBlank() ? statusLabel(
                available,
                gatePassed,
                healthy,
                issueDetected,
                warningCount,
                errorCount) : status.trim();
        issueCodes = issueCodes == null ? List.of() : List.copyOf(issueCodes);
    }

    public static TrainingReportDataHealthSummary from(TrainingReportDataHealth dataHealth) {
        TrainingReportDataHealth resolved = dataHealth == null
                ? TrainingReportDataHealth.fromMap(Map.of())
                : dataHealth;
        return new TrainingReportDataHealthSummary(
                resolved.available(),
                resolved.gatePassed(),
                resolved.healthy(),
                resolved.issueDetected(),
                resolved.issueCount(),
                resolved.warningCount(),
                resolved.errorCount(),
                null,
                resolved.issueCodes());
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("available", available);
        map.put("gatePassed", gatePassed);
        map.put("healthy", healthy);
        map.put("issueDetected", issueDetected);
        map.put("issueCount", issueCount);
        map.put("warningCount", warningCount);
        map.put("errorCount", errorCount);
        map.put("status", status);
        map.put("issueCodes", issueCodes);
        return Map.copyOf(map);
    }

    private static String statusLabel(
            boolean available,
            boolean gatePassed,
            boolean healthy,
            boolean issueDetected,
            int warningCount,
            int errorCount) {
        if (!available) {
            return "not-recorded";
        }
        if (errorCount > 0 || !gatePassed) {
            return "error";
        }
        if (warningCount > 0 || issueDetected) {
            return "warning";
        }
        return healthy ? "healthy" : "unknown";
    }
}
