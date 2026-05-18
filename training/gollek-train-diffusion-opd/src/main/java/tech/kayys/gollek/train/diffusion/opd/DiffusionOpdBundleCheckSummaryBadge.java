package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed nested badge payload for summarized health checks.
 */
public record DiffusionOpdBundleCheckSummaryBadge(
        String status,
        String alertLevel,
        String primaryIssueCode,
        String primaryCheckName,
        String primaryCheckMessage,
        String primaryCheckSeverity,
        double score,
        String variant,
        String label,
        String token,
        String checkStatus,
        String summaryMessage,
        String recommendedAction,
        String tooltip) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("status", status);
        map.put("alertLevel", alertLevel);
        map.put("primaryIssueCode", primaryIssueCode);
        map.put("primaryCheckName", primaryCheckName);
        map.put("primaryCheckMessage", primaryCheckMessage);
        map.put("primaryCheckSeverity", primaryCheckSeverity);
        map.put("score", score);
        map.put("variant", variant);
        map.put("label", label);
        map.put("token", token);
        map.put("checkStatus", checkStatus);
        map.put("summaryMessage", summaryMessage);
        map.put("recommendedAction", recommendedAction);
        map.put("tooltip", tooltip);
        return Map.copyOf(map);
    }
}
