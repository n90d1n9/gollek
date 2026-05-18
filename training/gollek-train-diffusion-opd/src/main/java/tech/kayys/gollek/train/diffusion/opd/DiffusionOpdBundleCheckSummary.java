package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed summary of bundle health checks.
 */
public record DiffusionOpdBundleCheckSummary(
        int total,
        int passed,
        int failed,
        double passRate,
        double failureRate,
        String status,
        String alertLevel,
        String summaryMessage,
        String recommendedAction,
        List<String> issueCodes,
        String primaryIssueCode,
        DiffusionOpdBundleCheckSummaryBadge healthBadge,
        String dominantSeverity,
        boolean allPassed,
        boolean hasCriticalFailures,
        boolean hasWarningFailures,
        boolean hasInfoFailures,
        Map<String, Integer> failingSeverityCounts,
        String primaryFailingName,
        String primaryFailingCode,
        String primaryFailingSeverity,
        String primaryFailingMessage,
        List<String> failingNames,
        List<String> failingCodes,
        Map<String, Integer> severityCounts) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("total", total);
        map.put("passed", passed);
        map.put("failed", failed);
        map.put("passRate", passRate);
        map.put("failureRate", failureRate);
        map.put("status", status);
        map.put("alertLevel", alertLevel);
        map.put("summaryMessage", summaryMessage);
        map.put("recommendedAction", recommendedAction);
        map.put("issueCodes", issueCodes);
        map.put("primaryIssueCode", primaryIssueCode);
        map.put("healthBadge", healthBadge.toMap());
        map.put("dominantSeverity", dominantSeverity);
        map.put("allPassed", allPassed);
        map.put("hasCriticalFailures", hasCriticalFailures);
        map.put("hasWarningFailures", hasWarningFailures);
        map.put("hasInfoFailures", hasInfoFailures);
        map.put("failingSeverityCounts", failingSeverityCounts);
        map.put("primaryFailingName", primaryFailingName);
        map.put("primaryFailingCode", primaryFailingCode);
        map.put("primaryFailingSeverity", primaryFailingSeverity);
        map.put("primaryFailingMessage", primaryFailingMessage);
        map.put("failingNames", failingNames);
        map.put("failingCodes", failingCodes);
        map.put("severityCounts", severityCounts);
        return Map.copyOf(map);
    }
}
