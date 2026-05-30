package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed top-level summary of a DiffusionOPD export bundle.
 *
 * <p>This is the lighter-weight derived inspection view: it emphasizes
 * section/format composition and missing-file overview without the full health
 * check detail carried by {@link DiffusionOpdBundleHealth}.
 */
public record DiffusionOpdBundleSummary(
        String bundleType,
        String sourceReportPath,
        String outputDirectory,
        String createdAt,
        int totalFiles,
        int existingFileCount,
        int missingFileCount,
        int sectionCount,
        int formatCount,
        DiffusionOpdBundleGroupedCount largestSection,
        List<DiffusionOpdBundleGroupedCount> sections,
        DiffusionOpdBundleGroupedCount largestFormat,
        List<DiffusionOpdBundleGroupedCount> formats,
        String focus,
        boolean dominantOnly,
        List<String> missingFiles) {

    public Map<String, Object> toMap() {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("bundleType", bundleType);
        map.put("sourceReportPath", sourceReportPath);
        map.put("outputDirectory", outputDirectory);
        map.put("createdAt", createdAt);
        map.put("totalFiles", totalFiles);
        map.put("existingFileCount", existingFileCount);
        map.put("missingFileCount", missingFileCount);
        map.put("sectionCount", sectionCount);
        map.put("formatCount", formatCount);
        map.put("largestSection", largestSection == null ? Map.of() : largestSection.toMap());
        map.put("sections", sections.stream().map(DiffusionOpdBundleGroupedCount::toMap).toList());
        map.put("largestFormat", largestFormat == null ? Map.of() : largestFormat.toMap());
        map.put("formats", formats.stream().map(DiffusionOpdBundleGroupedCount::toMap).toList());
        map.put("focus", focus);
        map.put("dominantOnly", dominantOnly);
        if (!missingFiles.isEmpty()) {
            map.put("missingFiles", missingFiles);
        }
        return Map.copyOf(map);
    }
}
