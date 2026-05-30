package tech.kayys.gollek.train.diffusion.opd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Owns manifest health snapshot, missing-artifact aggregation, and check
 * metrics.
 *
 * <p>This helper is the low-level health engine for inspector manifests. It
 * computes status, issue codes, checks, and grouped missing-artifact data
 * before {@link DiffusionOpdReportInspectorManifestHealthViews} shapes those
 * results for public output.
 */
final class DiffusionOpdReportInspectorManifestHealth {

    private DiffusionOpdReportInspectorManifestHealth() {
    }

    static ManifestHealthSnapshot collectManifestHealthSnapshot(
            DiffusionOpdBundleManifest manifest,
            Map<String, String> options) {
        List<Map<String, Object>> files = DiffusionOpdReportInspectorManifestFiles.filterManifestFilesList(manifest, "");
        List<Map<String, Object>> missingEntries = findMissingManifestFileEntries(manifest, files);
        List<String> missingFiles = missingEntries.stream()
                .map(entry -> String.valueOf(entry.getOrDefault("name", "")))
                .toList();
        String outputDirectory = manifest.outputDirectory();
        boolean outputDirectoryConfigured = !outputDirectory.isBlank();
        boolean outputDirectoryExists = outputDirectoryConfigured && Files.isDirectory(Path.of(outputDirectory));
        int totalFiles = files.size();
        int missingFileCount = missingFiles.size();
        int existingFileCount = totalFiles - missingFileCount;
        boolean healthy = missingFiles.isEmpty();
        double healthScore = totalFiles == 0 ? 1.0d : Math.max(0.0d, (double) existingFileCount / (double) totalFiles);
        String healthStatus = classifyBundleHealth(totalFiles, missingFileCount);
        String summaryMessage = buildBundleHealthSummary(healthStatus, totalFiles, missingFileCount);
        String recommendedAction = recommendedBundleHealthAction(healthStatus);
        List<String> issueCodes = bundleHealthIssueCodes(healthStatus, missingFileCount);
        String primaryIssueCode = issueCodes.isEmpty() ? "bundle.health.unknown" : issueCodes.get(0);
        String alertLevel = bundleHealthAlertLevel(healthStatus);
        List<Map<String, Object>> checks = buildBundleHealthChecks(
                outputDirectoryConfigured,
                outputDirectoryExists,
                totalFiles,
                missingFileCount,
                healthStatus);
        List<Map<String, Object>> failingChecks = checks.stream()
                .filter(check -> !Boolean.TRUE.equals(check.get("passed")))
                .toList();
        List<Map<String, Object>> missingSections = DiffusionOpdReportInspectorManifestFiles.limitManifestSummaryRows(
                summarizeMissingEntriesByField(missingEntries, "section"),
                options);
        List<Map<String, Object>> missingFormats = DiffusionOpdReportInspectorManifestFiles.limitManifestSummaryRows(
                summarizeMissingEntriesByField(missingEntries, "format"),
                options);
        return new ManifestHealthSnapshot(
                outputDirectory,
                outputDirectoryConfigured,
                outputDirectoryExists,
                totalFiles,
                existingFileCount,
                missingFileCount,
                healthy,
                healthScore,
                healthStatus,
                alertLevel,
                issueCodes,
                primaryIssueCode,
                summaryMessage,
                recommendedAction,
                checks,
                failingChecks,
                missingFiles,
                missingSections,
                missingFormats);
    }

    static CheckMetrics collectCheckMetrics(ManifestHealthSnapshot snapshot) {
        int passingCheckCount = snapshot.checks().size() - snapshot.failingChecks().size();
        int criticalCheckCount = countChecksBySeverity(snapshot.checks(), "critical");
        int warningCheckCount = countChecksBySeverity(snapshot.checks(), "warning");
        int infoCheckCount = countChecksBySeverity(snapshot.checks(), "info");
        int failingCriticalCheckCount = countChecksBySeverity(snapshot.failingChecks(), "critical");
        int failingWarningCheckCount = countChecksBySeverity(snapshot.failingChecks(), "warning");
        int failingInfoCheckCount = countChecksBySeverity(snapshot.failingChecks(), "info");
        String primaryFailingCheckName = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("name", ""));
        String primaryFailingCheckCode = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("code", ""));
        String primaryFailingCheckSeverity = snapshot.failingChecks().isEmpty()
                ? "none"
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("severity", ""));
        String primaryFailingCheckMessage = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("message", ""));
        return new CheckMetrics(
                passingCheckCount,
                criticalCheckCount,
                warningCheckCount,
                infoCheckCount,
                failingCriticalCheckCount,
                failingWarningCheckCount,
                failingInfoCheckCount,
                primaryFailingCheckName,
                primaryFailingCheckCode,
                primaryFailingCheckSeverity,
                primaryFailingCheckMessage,
                dominantCheckSeverity(criticalCheckCount, warningCheckCount, infoCheckCount));
    }

    static List<String> findMissingManifestFiles(DiffusionOpdBundleManifest manifest, List<Map<String, Object>> files) {
        return findMissingManifestFileEntries(manifest, files).stream()
                .map(entry -> String.valueOf(entry.getOrDefault("name", "")))
                .toList();
    }

    static String bundleHealthAlertLevel(String status) {
        return switch (status) {
            case "healthy" -> "info";
            case "broken" -> "critical";
            default -> "warning";
        };
    }

    static boolean hasFailingChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return checks.stream()
                .anyMatch(check -> !Boolean.TRUE.equals(check.get("passed"))
                        && severity.equalsIgnoreCase(String.valueOf(check.getOrDefault("severity", ""))));
    }

    private static List<Map<String, Object>> findMissingManifestFileEntries(
            DiffusionOpdBundleManifest manifest,
            List<Map<String, Object>> files) {
        if (manifest.outputDirectory().isBlank()) {
            return List.of();
        }
        Path outputDirectoryPath = Path.of(manifest.outputDirectory());
        List<Map<String, Object>> missing = new ArrayList<>();
        for (Map<String, Object> file : files) {
            Object name = file.get("name");
            if (name == null || String.valueOf(name).isBlank()) {
                continue;
            }
            if (!Files.exists(outputDirectoryPath.resolve(String.valueOf(name)))) {
                missing.add(file);
            }
        }
        return List.copyOf(missing);
    }

    private static List<Map<String, Object>> summarizeMissingEntriesByField(
            List<Map<String, Object>> missingEntries,
            String field) {
        LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
        for (Map<String, Object> entry : missingEntries) {
            String value = String.valueOf(entry.getOrDefault(field, ""));
            counts.merge(value, 1, Integer::sum);
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<>();
            row.put("field", field);
            row.put("value", entry.getKey());
            row.put("count", entry.getValue());
            rows.add(Map.copyOf(row));
        }
        rows.sort(Comparator.comparing(row -> ((Number) row.getOrDefault("count", 0)).intValue(), Comparator.reverseOrder()));
        return List.copyOf(rows);
    }

    private static String classifyBundleHealth(int totalFiles, int missingFileCount) {
        if (missingFileCount <= 0) {
            return "healthy";
        }
        if (missingFileCount >= totalFiles && totalFiles > 0) {
            return "broken";
        }
        return "degraded";
    }

    private static String buildBundleHealthSummary(String status, int totalFiles, int missingFileCount) {
        return switch (status) {
            case "healthy" -> "Bundle is healthy; all " + totalFiles + " manifest files are present.";
            case "broken" -> "Bundle is broken; all " + missingFileCount + " manifest files are missing.";
            default -> "Bundle is degraded; " + missingFileCount + " of " + totalFiles + " manifest files are missing.";
        };
    }

    private static String recommendedBundleHealthAction(String status) {
        return switch (status) {
            case "healthy" -> "none";
            case "broken" -> "rebuild-bundle";
            default -> "repair-missing-artifacts";
        };
    }

    private static List<String> bundleHealthIssueCodes(String status, int missingFileCount) {
        if ("healthy".equals(status)) {
            return List.of("bundle.ok");
        }
        if ("broken".equals(status)) {
            return List.of("bundle.artifacts.missing_all", "bundle.health.broken");
        }
        if (missingFileCount > 0) {
            return List.of("bundle.artifacts.missing_partial", "bundle.health.degraded");
        }
        return List.of("bundle.health.unknown");
    }

    private static List<Map<String, Object>> buildBundleHealthChecks(
            boolean outputDirectoryConfigured,
            boolean outputDirectoryExists,
            int totalFiles,
            int missingFileCount,
            String status) {
        List<Map<String, Object>> checks = new ArrayList<>();
        checks.add(healthCheck(
                "outputDirectory",
                outputDirectoryConfigured && outputDirectoryExists,
                outputDirectoryConfigured ? (outputDirectoryExists ? "info" : "critical") : "warning",
                outputDirectoryConfigured
                        ? (outputDirectoryExists ? "bundle.output_directory.ok" : "bundle.output_directory.missing")
                        : "bundle.output_directory.unset",
                outputDirectoryConfigured
                        ? (outputDirectoryExists ? "Output directory exists." : "Output directory is missing.")
                        : "Output directory is not configured."));
        checks.add(healthCheck(
                "artifactsComplete",
                missingFileCount == 0,
                bundleHealthAlertLevel(status),
                missingFileCount == 0
                        ? "bundle.artifacts.complete"
                        : (missingFileCount >= totalFiles && totalFiles > 0
                                ? "bundle.artifacts.missing_all"
                                : "bundle.artifacts.missing_partial"),
                missingFileCount == 0
                        ? "All manifest artifacts are present."
                        : (missingFileCount >= totalFiles && totalFiles > 0
                                ? "All manifest artifacts are missing."
                                : "Some manifest artifacts are missing.")));
        return List.copyOf(checks);
    }

    private static Map<String, Object> healthCheck(
            String name,
            boolean passed,
            String severity,
            String code,
            String message) {
        LinkedHashMap<String, Object> check = new LinkedHashMap<>();
        check.put("name", name);
        check.put("passed", passed);
        check.put("severity", severity);
        check.put("code", code);
        check.put("message", message);
        return Map.copyOf(check);
    }

    private static int countChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return (int) checks.stream()
                .filter(check -> severity.equalsIgnoreCase(String.valueOf(check.getOrDefault("severity", ""))))
                .count();
    }

    private static String dominantCheckSeverity(int criticalCheckCount, int warningCheckCount, int infoCheckCount) {
        if (criticalCheckCount >= warningCheckCount && criticalCheckCount >= infoCheckCount && criticalCheckCount > 0) {
            return "critical";
        }
        if (warningCheckCount >= infoCheckCount && warningCheckCount > 0) {
            return "warning";
        }
        if (infoCheckCount > 0) {
            return "info";
        }
        return "none";
    }

    record ManifestHealthSnapshot(
            String outputDirectory,
            boolean outputDirectoryConfigured,
            boolean outputDirectoryExists,
            int totalFiles,
            int existingFileCount,
            int missingFileCount,
            boolean healthy,
            double healthScore,
            String healthStatus,
            String alertLevel,
            List<String> issueCodes,
            String primaryIssueCode,
            String summaryMessage,
            String recommendedAction,
            List<Map<String, Object>> checks,
            List<Map<String, Object>> failingChecks,
            List<String> missingFiles,
            List<Map<String, Object>> missingSections,
            List<Map<String, Object>> missingFormats) {
    }

    record CheckMetrics(
            int passingCheckCount,
            int criticalCheckCount,
            int warningCheckCount,
            int infoCheckCount,
            int failingCriticalCheckCount,
            int failingWarningCheckCount,
            int failingInfoCheckCount,
            String primaryFailingCheckName,
            String primaryFailingCheckCode,
            String primaryFailingCheckSeverity,
            String primaryFailingCheckMessage,
            String dominantSeverity) {
    }
}
