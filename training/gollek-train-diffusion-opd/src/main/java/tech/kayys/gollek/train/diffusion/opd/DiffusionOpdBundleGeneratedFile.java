package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Typed generated-file entry for exported DiffusionOPD bundle manifests.
 */
public record DiffusionOpdBundleGeneratedFile(
        String name,
        String section,
        String format,
        String entryFormat,
        String columns,
        String entryColumns) {

    /**
     * Adapts one raw generated-file manifest entry into the typed generated-file record.
     */
    public static DiffusionOpdBundleGeneratedFile fromMap(Map<String, Object> map) {
        return new DiffusionOpdBundleGeneratedFile(
                stringValue(map, "name"),
                stringValue(map, "section"),
                stringValue(map, "format"),
                stringValue(map, "entryFormat"),
                stringValue(map, "columns"),
                stringValue(map, "entryColumns"));
    }

    /**
     * Converts this typed generated-file record back into the normalized manifest-entry map shape.
     */
    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> value = new LinkedHashMap<>();
        value.put("name", name);
        value.put("section", section);
        value.put("format", format);
        value.put("entryFormat", entryFormat);
        value.put("columns", columns);
        value.put("entryColumns", entryColumns);
        return Map.copyOf(value);
    }

    private static String stringValue(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value == null ? "" : String.valueOf(value);
    }
}
