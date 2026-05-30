package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Owns manifest health badges, check summaries, and typed health view mapping.
 *
 * <p>This helper sits above {@link DiffusionOpdReportInspectorManifestHealth}:
 * it takes the computed health snapshot and turns it into CLI-facing maps and
 * typed API view objects.
 */
final class DiffusionOpdReportInspectorManifestHealthViews {

    private DiffusionOpdReportInspectorManifestHealthViews() {
    }

    static void applyManifestHealthRoot(
            LinkedHashMap<String, Object> health,
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot) {
        health.put("outputDirectory", snapshot.outputDirectory());
        health.put("outputDirectoryConfigured", snapshot.outputDirectoryConfigured());
        health.put("outputDirectoryExists", snapshot.outputDirectoryExists());
        health.put("totalFiles", snapshot.totalFiles());
        health.put("existingFileCount", snapshot.existingFileCount());
        health.put("missingFileCount", snapshot.missingFileCount());
        health.put("healthy", snapshot.healthy());
        health.put("status", snapshot.healthStatus());
        health.put("healthScore", snapshot.healthScore());
        health.put("alertLevel", snapshot.alertLevel());
        health.put("issueCodes", snapshot.issueCodes());
        health.put("primaryIssueCode", snapshot.primaryIssueCode());
        health.put("healthBadge", buildBundleHealthBadge(snapshot));
        health.put("summaryMessage", snapshot.summaryMessage());
        health.put("recommendedAction", snapshot.recommendedAction());
        health.put("checks", snapshot.checks());
    }

    static void applyManifestHealthChecks(
            LinkedHashMap<String, Object> health,
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot) {
        DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics =
                DiffusionOpdReportInspectorManifestHealth.collectCheckMetrics(snapshot);
        health.put("healthBadge", buildBundleHealthBadge(snapshot, metrics));
        health.put("failingChecks", snapshot.failingChecks());
        health.put("failingCheckCount", snapshot.failingChecks().size());
        health.put("passingCheckCount", metrics.passingCheckCount());
        health.put("criticalCheckCount", metrics.criticalCheckCount());
        health.put("warningCheckCount", metrics.warningCheckCount());
        health.put("infoCheckCount", metrics.infoCheckCount());
        health.put("checkSummary", buildCheckSummary(snapshot, metrics));
    }

    static DiffusionOpdBundleHealth toTypedBundleHealth(
            DiffusionOpdBundleManifest manifest,
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics) {
        return new DiffusionOpdBundleHealth(
                manifest.bundleType(),
                snapshot.outputDirectory(),
                snapshot.outputDirectoryConfigured(),
                snapshot.outputDirectoryExists(),
                snapshot.totalFiles(),
                snapshot.existingFileCount(),
                snapshot.missingFileCount(),
                snapshot.healthy(),
                snapshot.healthStatus(),
                snapshot.healthScore(),
                snapshot.alertLevel(),
                snapshot.issueCodes(),
                snapshot.primaryIssueCode(),
                toTypedBundleHealthBadge(snapshot, metrics),
                snapshot.summaryMessage(),
                snapshot.recommendedAction(),
                snapshot.checks().stream()
                        .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedBundleHealthCheck)
                        .toList(),
                snapshot.failingChecks().stream()
                        .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedBundleHealthCheck)
                        .toList(),
                snapshot.failingChecks().size(),
                metrics.passingCheckCount(),
                metrics.criticalCheckCount(),
                metrics.warningCheckCount(),
                metrics.infoCheckCount(),
                toTypedBundleCheckSummary(snapshot, metrics),
                snapshot.missingFiles(),
                snapshot.missingSections().stream()
                        .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedGroupedCount)
                        .toList(),
                snapshot.missingFormats().stream()
                        .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedGroupedCount)
                        .toList());
    }

    static DiffusionOpdBundleGroupedCount toTypedGroupedCount(Map<String, Object> row) {
        Object countValue = row.get("count");
        int count = countValue instanceof Number number ? number.intValue() : 0;
        Object namesValue = row.get("names");
        List<String> names = namesValue instanceof List<?> list
                ? list.stream().map(String::valueOf).toList()
                : List.of();
        return new DiffusionOpdBundleGroupedCount(
                String.valueOf(row.getOrDefault("groupBy", "")),
                String.valueOf(row.getOrDefault("value", "")),
                count,
                names);
    }

    private static Map<String, Object> buildBundleHealthBadge(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot) {
        return Map.of(
                "status", snapshot.healthStatus(),
                "alertLevel", snapshot.alertLevel(),
                "primaryIssueCode", snapshot.primaryIssueCode(),
                "score", snapshot.healthScore(),
                "variant", bundleHealthBadgeVariant(snapshot.healthStatus()),
                "label", bundleHealthBadgeLabel(snapshot.healthStatus()),
                "token", bundleHealthBadgeToken(snapshot.healthStatus()),
                "tooltip", snapshot.summaryMessage(),
                "checkStatus", "pending");
    }

    private static DiffusionOpdBundleHealthBadge toTypedBundleHealthBadge(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics) {
        return new DiffusionOpdBundleHealthBadge(
                snapshot.healthStatus(),
                snapshot.alertLevel(),
                snapshot.primaryIssueCode(),
                metrics.primaryFailingCheckCode(),
                metrics.primaryFailingCheckSeverity(),
                snapshot.healthScore(),
                bundleHealthBadgeVariant(snapshot.healthStatus()),
                bundleHealthBadgeLabel(snapshot.healthStatus()),
                bundleHealthBadgeToken(snapshot.healthStatus()),
                snapshot.summaryMessage(),
                snapshot.failingChecks().isEmpty() ? "pass" : "fail");
    }

    private static Map<String, Object> buildBundleHealthBadge(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics) {
        LinkedHashMap<String, Object> badge = new LinkedHashMap<>();
        badge.put("status", snapshot.healthStatus());
        badge.put("alertLevel", snapshot.alertLevel());
        badge.put("primaryIssueCode", snapshot.primaryIssueCode());
        badge.put("primaryCheckCode", metrics.primaryFailingCheckCode());
        badge.put("primaryCheckSeverity", metrics.primaryFailingCheckSeverity());
        badge.put("score", snapshot.healthScore());
        badge.put("variant", bundleHealthBadgeVariant(snapshot.healthStatus()));
        badge.put("label", bundleHealthBadgeLabel(snapshot.healthStatus()));
        badge.put("token", bundleHealthBadgeToken(snapshot.healthStatus()));
        badge.put("tooltip", snapshot.summaryMessage());
        badge.put("checkStatus", snapshot.failingChecks().isEmpty() ? "pass" : "fail");
        return badge;
    }

    private static Map<String, Object> buildCheckSummary(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics) {
        String checkStatus = snapshot.failingChecks().isEmpty() ? "healthy" : snapshot.healthStatus();
        String checkAlertLevel = snapshot.failingChecks().isEmpty()
                ? "info"
                : DiffusionOpdReportInspectorManifestHealth.bundleHealthAlertLevel(snapshot.healthStatus());
        String checkSummaryMessage = snapshot.failingChecks().isEmpty()
                ? "All bundle health checks passed."
                : "Bundle health checks reported " + snapshot.failingChecks().size() + " failure(s).";
        String checkRecommendedAction = snapshot.failingChecks().isEmpty()
                ? "none"
                : ("critical".equalsIgnoreCase(metrics.dominantSeverity())
                        ? "repair-critical-checks"
                        : "review-failing-checks");
        List<String> checkIssueCodes = snapshot.failingChecks().isEmpty()
                ? List.of("checks.ok")
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .distinct()
                        .toList();
        String primaryCheckIssueCode = snapshot.failingChecks().isEmpty()
                ? "checks.ok"
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .findFirst()
                        .orElse("checks.unknown");
        Map<String, Object> nestedBadge = buildCheckSummaryBadge(
                snapshot,
                metrics,
                checkStatus,
                checkAlertLevel,
                primaryCheckIssueCode,
                checkSummaryMessage,
                checkRecommendedAction);
        LinkedHashMap<String, Object> summary = new LinkedHashMap<>();
        summary.put("total", snapshot.checks().size());
        summary.put("passed", metrics.passingCheckCount());
        summary.put("failed", snapshot.failingChecks().size());
        summary.put("passRate", snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size());
        summary.put("failureRate", snapshot.checks().isEmpty() ? 0.0d : (double) snapshot.failingChecks().size() / (double) snapshot.checks().size());
        summary.put("status", checkStatus);
        summary.put("alertLevel", checkAlertLevel);
        summary.put("summaryMessage", checkSummaryMessage);
        summary.put("recommendedAction", checkRecommendedAction);
        summary.put("issueCodes", checkIssueCodes);
        summary.put("primaryIssueCode", primaryCheckIssueCode);
        summary.put("healthBadge", nestedBadge);
        summary.put("dominantSeverity", metrics.dominantSeverity());
        summary.put("allPassed", snapshot.failingChecks().isEmpty());
        summary.put("hasCriticalFailures", DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "critical"));
        summary.put("hasWarningFailures", DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "warning"));
        summary.put("hasInfoFailures", DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "info"));
        summary.put("failingSeverityCounts", Map.of(
                "critical", metrics.failingCriticalCheckCount(),
                "warning", metrics.failingWarningCheckCount(),
                "info", metrics.failingInfoCheckCount()));
        summary.put("primaryFailingName", metrics.primaryFailingCheckName());
        summary.put("primaryFailingCode", metrics.primaryFailingCheckCode());
        summary.put("primaryFailingSeverity", metrics.primaryFailingCheckSeverity());
        summary.put("primaryFailingMessage", metrics.primaryFailingCheckMessage());
        summary.put("failingNames", snapshot.failingChecks().stream()
                .map(check -> String.valueOf(check.getOrDefault("name", "")))
                .toList());
        summary.put("failingCodes", snapshot.failingChecks().stream()
                .map(check -> String.valueOf(check.getOrDefault("code", "")))
                .toList());
        summary.put("severityCounts", Map.of(
                "critical", metrics.criticalCheckCount(),
                "warning", metrics.warningCheckCount(),
                "info", metrics.infoCheckCount()));
        return summary;
    }

    private static DiffusionOpdBundleCheckSummary toTypedBundleCheckSummary(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics) {
        String checkStatus = snapshot.failingChecks().isEmpty() ? "healthy" : snapshot.healthStatus();
        String checkAlertLevel = snapshot.failingChecks().isEmpty()
                ? "info"
                : DiffusionOpdReportInspectorManifestHealth.bundleHealthAlertLevel(snapshot.healthStatus());
        String checkSummaryMessage = snapshot.failingChecks().isEmpty()
                ? "All bundle health checks passed."
                : "Bundle health checks reported " + snapshot.failingChecks().size() + " failure(s).";
        String checkRecommendedAction = snapshot.failingChecks().isEmpty()
                ? "none"
                : ("critical".equalsIgnoreCase(metrics.dominantSeverity())
                        ? "repair-critical-checks"
                        : "review-failing-checks");
        List<String> checkIssueCodes = snapshot.failingChecks().isEmpty()
                ? List.of("checks.ok")
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .distinct()
                        .toList();
        String primaryCheckIssueCode = snapshot.failingChecks().isEmpty()
                ? "checks.ok"
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .findFirst()
                        .orElse("checks.unknown");
        return new DiffusionOpdBundleCheckSummary(
                snapshot.checks().size(),
                metrics.passingCheckCount(),
                snapshot.failingChecks().size(),
                snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size(),
                snapshot.checks().isEmpty() ? 0.0d : (double) snapshot.failingChecks().size() / (double) snapshot.checks().size(),
                checkStatus,
                checkAlertLevel,
                checkSummaryMessage,
                checkRecommendedAction,
                checkIssueCodes,
                primaryCheckIssueCode,
                toTypedBundleCheckSummaryBadge(
                        snapshot,
                        metrics,
                        checkStatus,
                        checkAlertLevel,
                        primaryCheckIssueCode,
                        checkSummaryMessage,
                        checkRecommendedAction),
                metrics.dominantSeverity(),
                snapshot.failingChecks().isEmpty(),
                DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "critical"),
                DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "warning"),
                DiffusionOpdReportInspectorManifestHealth.hasFailingChecksBySeverity(snapshot.failingChecks(), "info"),
                Map.of(
                        "critical", metrics.failingCriticalCheckCount(),
                        "warning", metrics.failingWarningCheckCount(),
                        "info", metrics.failingInfoCheckCount()),
                metrics.primaryFailingCheckName(),
                metrics.primaryFailingCheckCode(),
                metrics.primaryFailingCheckSeverity(),
                metrics.primaryFailingCheckMessage(),
                snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("name", "")))
                        .toList(),
                snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .toList(),
                Map.of(
                        "critical", metrics.criticalCheckCount(),
                        "warning", metrics.warningCheckCount(),
                        "info", metrics.infoCheckCount()));
    }

    private static Map<String, Object> buildCheckSummaryBadge(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics,
            String checkStatus,
            String checkAlertLevel,
            String primaryCheckIssueCode,
            String checkSummaryMessage,
            String checkRecommendedAction) {
        LinkedHashMap<String, Object> badge = new LinkedHashMap<>();
        badge.put("status", checkStatus);
        badge.put("alertLevel", checkAlertLevel);
        badge.put("primaryIssueCode", primaryCheckIssueCode);
        badge.put("primaryCheckName", metrics.primaryFailingCheckName());
        badge.put("primaryCheckMessage", metrics.primaryFailingCheckMessage());
        badge.put("primaryCheckSeverity", metrics.primaryFailingCheckSeverity());
        badge.put("score", snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size());
        badge.put("variant", bundleHealthBadgeVariant(checkStatus));
        badge.put("label", bundleHealthBadgeLabel(checkStatus));
        badge.put("token", bundleHealthBadgeToken(checkStatus));
        badge.put("checkStatus", snapshot.failingChecks().isEmpty() ? "pass" : "fail");
        badge.put("summaryMessage", checkSummaryMessage);
        badge.put("recommendedAction", checkRecommendedAction);
        badge.put("tooltip", checkSummaryMessage);
        return badge;
    }

    private static DiffusionOpdBundleCheckSummaryBadge toTypedBundleCheckSummaryBadge(
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics,
            String checkStatus,
            String checkAlertLevel,
            String primaryCheckIssueCode,
            String checkSummaryMessage,
            String checkRecommendedAction) {
        return new DiffusionOpdBundleCheckSummaryBadge(
                checkStatus,
                checkAlertLevel,
                primaryCheckIssueCode,
                metrics.primaryFailingCheckName(),
                metrics.primaryFailingCheckMessage(),
                metrics.primaryFailingCheckSeverity(),
                snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size(),
                bundleHealthBadgeVariant(checkStatus),
                bundleHealthBadgeLabel(checkStatus),
                bundleHealthBadgeToken(checkStatus),
                snapshot.failingChecks().isEmpty() ? "pass" : "fail",
                checkSummaryMessage,
                checkRecommendedAction,
                checkSummaryMessage);
    }

    private static DiffusionOpdBundleHealthCheck toTypedBundleHealthCheck(Map<String, Object> check) {
        return new DiffusionOpdBundleHealthCheck(
                String.valueOf(check.getOrDefault("name", "")),
                Boolean.TRUE.equals(check.get("passed")),
                String.valueOf(check.getOrDefault("severity", "")),
                String.valueOf(check.getOrDefault("code", "")),
                String.valueOf(check.getOrDefault("message", "")));
    }

    private static String bundleHealthBadgeVariant(String status) {
        return switch (status) {
            case "healthy" -> "success";
            case "broken" -> "danger";
            default -> "warning";
        };
    }

    private static String bundleHealthBadgeLabel(String status) {
        return switch (status) {
            case "healthy" -> "Healthy";
            case "broken" -> "Broken";
            default -> "Degraded";
        };
    }

    private static String bundleHealthBadgeToken(String status) {
        return switch (status) {
            case "healthy" -> "bundle-health-healthy";
            case "broken" -> "bundle-health-broken";
            default -> "bundle-health-degraded";
        };
    }
}
