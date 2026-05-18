package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed badge payload for top-level bundle health.
 */
public record DiffusionOpdBundleHealthBadge(
        String status,
        String alertLevel,
        String primaryIssueCode,
        String primaryCheckCode,
        String primaryCheckSeverity,
        double score,
        String variant,
        String label,
        String token,
        String tooltip,
        String checkStatus) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("status", status);
        map.put("alertLevel", alertLevel);
        map.put("primaryIssueCode", primaryIssueCode);
        map.put("primaryCheckCode", primaryCheckCode);
        map.put("primaryCheckSeverity", primaryCheckSeverity);
        map.put("score", score);
        map.put("variant", variant);
        map.put("label", label);
        map.put("token", token);
        map.put("tooltip", tooltip);
        map.put("checkStatus", checkStatus);
        return Map.copyOf(map);
    }
}
