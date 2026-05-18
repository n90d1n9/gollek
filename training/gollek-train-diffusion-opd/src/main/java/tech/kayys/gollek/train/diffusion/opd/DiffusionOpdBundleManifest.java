package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed bundle manifest model used by the DiffusionOPD report inspector.
 */
public record DiffusionOpdBundleManifest(
        String bundleType,
        String sourceReportPath,
        String outputDirectory,
        String createdAt,
        String columnStrategy,
        List<DiffusionOpdBundleGeneratedFile> generatedFiles,
        Map<String, Object> raw) {

    public static DiffusionOpdBundleManifest fromMap(Map<String, Object> raw) {
        return new DiffusionOpdBundleManifest(
                stringValue(raw, "bundleType"),
                stringValue(raw, "sourceReportPath"),
                stringValue(raw, "outputDirectory"),
                stringValue(raw, "createdAt"),
                stringValue(raw, "columnStrategy"),
                generatedFiles(raw.get("generatedFiles")),
                Map.copyOf(new LinkedHashMap<>(raw)));
    }

    public Map<String, Object> toMap() {
        return raw;
    }

    private static List<DiffusionOpdBundleGeneratedFile> generatedFiles(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        return list.stream()
                .map(DiffusionOpdBundleManifest::generatedFileEntry)
                .map(DiffusionOpdBundleGeneratedFile::fromMap)
                .toList();
    }

    private static Map<String, Object> generatedFileEntry(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        LinkedHashMap<String, Object> normalized = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            normalized.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(normalized);
    }

    private static String stringValue(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value == null ? "" : String.valueOf(value);
    }
}
