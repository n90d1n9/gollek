package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed bundle health check entry.
 */
public record DiffusionOpdBundleHealthCheck(
        String name,
        boolean passed,
        String severity,
        String code,
        String message) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("name", name);
        map.put("passed", passed);
        map.put("severity", severity);
        map.put("code", code);
        map.put("message", message);
        return Map.copyOf(map);
    }
}
