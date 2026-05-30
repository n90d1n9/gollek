package tech.kayys.gollek.train.diffusion.opd;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Owns manifest bundle-summary and grouped file-summary shaping.
 *
 * <p>This helper is responsible for higher-level manifest summaries after file
 * lookups and health calculations have already been resolved by the narrower
 * manifest helpers.
 */
final class DiffusionOpdReportInspectorManifestSummaries {

    private DiffusionOpdReportInspectorManifestSummaries() {
    }

    static Map<String, Object> summarizeManifestBundle(DiffusionOpdBundleManifest manifest) {
        return summarizeManifestBundle(manifest, "");
    }

    static DiffusionOpdBundleSummary inspectManifestSummary(DiffusionOpdBundleManifest manifest) {
        return toTypedBundleSummary(manifest, Map.of());
    }

    static Map<String, Object> summarizeManifestBundle(DiffusionOpdBundleManifest manifest, String optionsExpression) {
        Map<String, String> options = DiffusionOpdReportInspectorManifestFiles.parseManifestFilters(optionsExpression);
        LinkedHashMap<String, Object> summary = new LinkedHashMap<>(toTypedBundleSummary(manifest, options).toMap());
        if (options.containsKey("top")) {
            summary.put("top", options.get("top"));
        }
        return Map.copyOf(summary);
    }

    static Object summarizeManifestFiles(DiffusionOpdBundleManifest manifest, String filterExpression) {
        var query = DiffusionOpdReportInspectorManifestFiles.parseManifestFileQuery(filterExpression);
        Map<String, String> filters = query.filters();
        String groupBy = query.groupBy();
        List<Map<String, Object>> matches =
                DiffusionOpdReportInspectorManifestFiles.filterManifestFilesList(manifest, filterExpression);
        LinkedHashMap<String, LinkedHashMap<String, Object>> grouped = new LinkedHashMap<>();
        for (Map<String, Object> match : matches) {
            String value = String.valueOf(match.getOrDefault(groupBy, ""));
            LinkedHashMap<String, Object> summary = grouped.computeIfAbsent(value, ignored -> {
                LinkedHashMap<String, Object> created = new LinkedHashMap<>();
                created.put("groupBy", groupBy);
                created.put("value", value);
                created.put("count", 0);
                created.put("names", new ArrayList<String>());
                return created;
            });
            summary.put("count", ((Integer) summary.get("count")) + 1);
            @SuppressWarnings("unchecked")
            List<String> names = (List<String>) summary.get("names");
            names.add(String.valueOf(match.getOrDefault("name", "")));
        }
        List<Map<String, Object>> rows = new ArrayList<>(grouped.values());
        DiffusionOpdReportInspectorManifestFiles.sortManifestSummaryRows(rows, filters);
        rows = DiffusionOpdReportInspectorManifestFiles.limitManifestSummaryRows(rows, filters);
        return List.copyOf(rows);
    }

    static List<Map<String, Object>> castSummaryRows(Object value) {
        if (value instanceof List<?> list) {
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object element : list) {
                if (element instanceof Map<?, ?> entry) {
                    rows.add(DiffusionOpdReportInspectorRendering.normalizeMap(entry));
                }
            }
            return List.copyOf(rows);
        }
        return List.of();
    }

    private static DiffusionOpdBundleSummary toTypedBundleSummary(
            DiffusionOpdBundleManifest manifest,
            Map<String, String> options) {
        List<Map<String, Object>> files =
                DiffusionOpdReportInspectorManifestFiles.filterManifestFilesList(manifest, "");
        List<Map<String, Object>> bySection = DiffusionOpdReportInspectorManifestFiles.limitManifestSummaryRows(
                castSummaryRows(summarizeManifestFiles(manifest, "by=section:sort=-count")),
                options);
        List<Map<String, Object>> byFormat = DiffusionOpdReportInspectorManifestFiles.limitManifestSummaryRows(
                castSummaryRows(summarizeManifestFiles(manifest, "by=format:sort=-count")),
                options);
        String focus = options.getOrDefault("focus", "all").trim().toLowerCase(Locale.ROOT);
        boolean dominantOnly = Boolean.parseBoolean(options.getOrDefault("dominant", "false"));
        List<String> missingFiles = DiffusionOpdReportInspectorManifestHealth.findMissingManifestFiles(manifest, files);
        List<DiffusionOpdBundleGroupedCount> typedSections = bySection.stream()
                .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedGroupedCount)
                .toList();
        List<DiffusionOpdBundleGroupedCount> typedFormats = byFormat.stream()
                .map(DiffusionOpdReportInspectorManifestHealthViews::toTypedGroupedCount)
                .toList();
        return new DiffusionOpdBundleSummary(
                manifest.bundleType(),
                manifest.sourceReportPath(),
                manifest.outputDirectory(),
                manifest.createdAt(),
                files.size(),
                files.size() - missingFiles.size(),
                missingFiles.size(),
                typedSections.size(),
                typedFormats.size(),
                typedSections.isEmpty() ? null : typedSections.getFirst(),
                "formats".equals(focus) || dominantOnly ? List.of() : typedSections,
                typedFormats.isEmpty() ? null : typedFormats.getFirst(),
                "sections".equals(focus) || dominantOnly ? List.of() : typedFormats,
                focus,
                dominantOnly,
                limitMissingFiles(missingFiles, options));
    }

    static List<String> limitMissingFiles(List<String> missingFiles, Map<String, String> options) {
        if (!options.containsKey("top")) {
            return missingFiles;
        }
        try {
            int top = Integer.parseInt(options.get("top"));
            if (top <= 0) {
                return List.of();
            }
            if (top < missingFiles.size()) {
                return List.copyOf(missingFiles.subList(0, top));
            }
        } catch (NumberFormatException ignored) {
        }
        return missingFiles;
    }
}
