package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed grouped-count row used by bundle summaries and health diagnostics.
 */
public record DiffusionOpdBundleGroupedCount(
        String groupBy,
        String value,
        int count,
        List<String> names) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("groupBy", groupBy);
        map.put("value", value);
        map.put("count", count);
        map.put("names", names);
        return Map.copyOf(map);
    }
}
