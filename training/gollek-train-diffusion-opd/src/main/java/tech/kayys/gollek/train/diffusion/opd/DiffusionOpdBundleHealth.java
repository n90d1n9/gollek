package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed bundle health report for exported DiffusionOPD bundles.
 */
public record DiffusionOpdBundleHealth(
        String bundleType,
        String outputDirectory,
        boolean outputDirectoryConfigured,
        boolean outputDirectoryExists,
        int totalFiles,
        int existingFileCount,
        int missingFileCount,
        boolean healthy,
        String status,
        double healthScore,
        String alertLevel,
        List<String> issueCodes,
        String primaryIssueCode,
        DiffusionOpdBundleHealthBadge healthBadge,
        String summaryMessage,
        String recommendedAction,
        List<DiffusionOpdBundleHealthCheck> checks,
        List<DiffusionOpdBundleHealthCheck> failingChecks,
        int failingCheckCount,
        int passingCheckCount,
        int criticalCheckCount,
        int warningCheckCount,
        int infoCheckCount,
        DiffusionOpdBundleCheckSummary checkSummary,
        List<String> missingFiles,
        List<DiffusionOpdBundleGroupedCount> missingSections,
        List<DiffusionOpdBundleGroupedCount> missingFormats) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("bundleType", bundleType);
        map.put("outputDirectory", outputDirectory);
        map.put("outputDirectoryConfigured", outputDirectoryConfigured);
        map.put("outputDirectoryExists", outputDirectoryExists);
        map.put("totalFiles", totalFiles);
        map.put("existingFileCount", existingFileCount);
        map.put("missingFileCount", missingFileCount);
        map.put("healthy", healthy);
        map.put("status", status);
        map.put("healthScore", healthScore);
        map.put("alertLevel", alertLevel);
        map.put("issueCodes", issueCodes);
        map.put("primaryIssueCode", primaryIssueCode);
        map.put("healthBadge", healthBadge.toMap());
        map.put("summaryMessage", summaryMessage);
        map.put("recommendedAction", recommendedAction);
        map.put("checks", checks.stream().map(DiffusionOpdBundleHealthCheck::toMap).toList());
        map.put("failingChecks", failingChecks.stream().map(DiffusionOpdBundleHealthCheck::toMap).toList());
        map.put("failingCheckCount", failingCheckCount);
        map.put("passingCheckCount", passingCheckCount);
        map.put("criticalCheckCount", criticalCheckCount);
        map.put("warningCheckCount", warningCheckCount);
        map.put("infoCheckCount", infoCheckCount);
        map.put("checkSummary", checkSummary.toMap());
        map.put("missingFiles", missingFiles);
        map.put("missingSections", missingSections.stream().map(DiffusionOpdBundleGroupedCount::toMap).toList());
        map.put("missingFormats", missingFormats.stream().map(DiffusionOpdBundleGroupedCount::toMap).toList());
        return Map.copyOf(map);
    }
}
