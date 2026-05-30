package tech.kayys.gollek.train.diffusion.opd;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Owns manifest file filtering, lookup, and query parsing for the inspector.
 *
 * <p>This helper stays at the generated-file query layer, while
 * {@link DiffusionOpdReportInspectorManifestSummaries} handles grouped summary
 * shaping and {@link DiffusionOpdReportInspectorManifestSections} owns section
 * routing.
 */
final class DiffusionOpdReportInspectorManifestFiles {

    private DiffusionOpdReportInspectorManifestFiles() {
    }

    static List<Map<String, Object>> filterManifestFilesList(
            DiffusionOpdBundleManifest manifest,
            String filterExpression) {
        ManifestFileQuery query = parseManifestFileQuery(filterExpression);
        Map<String, String> filters = query.filters();
        List<Map<String, Object>> matches = new ArrayList<>();
        for (Map<String, Object> normalized : manifestGeneratedFileEntries(manifest)) {
            if (matchesManifestFilters(normalized, filters)) {
                matches.add(normalized);
            }
        }
        sortManifestFiles(matches, filters);
        return List.copyOf(matches);
    }

    static List<Map<String, Object>> manifestGeneratedFileEntries(DiffusionOpdBundleManifest manifest) {
        return manifest.generatedFiles().stream()
                .map(DiffusionOpdBundleGeneratedFile::toMap)
                .toList();
    }

    static DiffusionOpdBundleGeneratedFile resolveTypedManifestFileEntry(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        Map<String, Object> entry = resolveManifestFileEntry(manifest, requested);
        return entry.isEmpty() ? null : DiffusionOpdBundleGeneratedFile.fromMap(entry);
    }

    static Map<String, String> parseManifestFilters(String filterExpression) {
        LinkedHashMap<String, String> filters = new LinkedHashMap<>();
        for (String token : filterExpression.split(":")) {
            int equals = token.indexOf('=');
            if (equals <= 0 || equals >= token.length() - 1) {
                continue;
            }
            String key = token.substring(0, equals).trim().toLowerCase(Locale.ROOT);
            String value = token.substring(equals + 1).trim();
            if (!key.isEmpty() && !value.isEmpty()) {
                filters.put(key, value);
            }
        }
        return Map.copyOf(filters);
    }

    static void sortManifestSummaryRows(List<Map<String, Object>> rows, Map<String, String> filters) {
        ManifestSortSpec sortSpec = parseManifestSortSpec(filters.get("sort"));
        if (sortSpec == null) {
            return;
        }
        Comparator<Map<String, Object>> comparator;
        if ("count".equals(sortSpec.field())) {
            comparator = Comparator.comparing(row -> ((Number) row.getOrDefault("count", 0)).intValue());
        } else {
            comparator = Comparator.comparing(
                    row -> String.valueOf(row.getOrDefault(sortSpec.field(), "")).toLowerCase(Locale.ROOT));
        }
        rows.sort(sortSpec.descending() ? comparator.reversed() : comparator);
    }

    static List<Map<String, Object>> limitManifestSummaryRows(
            List<Map<String, Object>> rows,
            Map<String, String> filters) {
        Integer top = parseOptionalInteger(filters.get("top"));
        if (top == null) {
            return rows;
        }
        if (top <= 0) {
            return List.of();
        }
        if (top >= rows.size()) {
            return rows;
        }
        return new ArrayList<>(rows.subList(0, top));
    }

    static Object findManifestFile(DiffusionOpdBundleManifest manifest, String requested) {
        return findManifestFileEntryByName(manifestGeneratedFileEntries(manifest), requested);
    }

    static Object loadManifestFileContent(DiffusionOpdBundleManifest manifest, String requested) {
        DiffusionOpdBundleGeneratedFile entry = resolveTypedManifestFileEntry(manifest, requested);
        if (entry == null) {
            return Map.of();
        }
        if (manifest.outputDirectory().isBlank() || entry.name().isBlank()) {
            return Map.of();
        }
        Path filePath = Path.of(manifest.outputDirectory()).resolve(entry.name());
        return DiffusionOpdReportInspectorSupport.loadBundleFileContent(filePath, entry.format());
    }

    static Object loadTypedBundleFileContent(
            DiffusionOpdBundleManifest manifest,
            DiffusionOpdBundleGeneratedFile entry) {
        if (manifest.outputDirectory().isBlank() || entry.name().isBlank()) {
            return Map.of();
        }
        Path filePath = Path.of(manifest.outputDirectory()).resolve(entry.name());
        return DiffusionOpdReportInspectorSupport.loadBundleFileContent(filePath, entry.format());
    }

    static ManifestFileQuery parseManifestFileQuery(String filterExpression) {
        Map<String, String> filters = parseManifestFilters(filterExpression);
        return new ManifestFileQuery(
                filters,
                normalizedManifestFilterValue(filters, "by", "section"),
                normalizedManifestFilterValue(filters, "pick", "first"),
                filters.get("sort"),
                parseManifestSortSpec(filters.get("sort")),
                parseOptionalInteger(filters.get("index")),
                parseOptionalInteger(filters.get("top")));
    }

    private static String normalizedManifestFilterValue(
            Map<String, String> filters,
            String key,
            String defaultValue) {
        return filters.getOrDefault(key, defaultValue).trim().toLowerCase(Locale.ROOT);
    }

    private static Integer parseOptionalInteger(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static ManifestSortSpec parseManifestSortSpec(String sort) {
        if (sort == null || sort.isBlank()) {
            return null;
        }
        boolean descending = sort.startsWith("-");
        String field = (descending ? sort.substring(1) : sort).trim().toLowerCase(Locale.ROOT);
        if (field.isEmpty()) {
            return null;
        }
        return new ManifestSortSpec(field, descending);
    }

    private static boolean matchesManifestFilters(Map<String, Object> file, Map<String, String> filters) {
        for (Map.Entry<String, String> filter : filters.entrySet()) {
            if (isManifestControlFilter(filter.getKey())) {
                continue;
            }
            Object actual = file.get(filter.getKey());
            if (actual == null || !filter.getValue().equalsIgnoreCase(String.valueOf(actual))) {
                return false;
            }
        }
        return true;
    }

    private static boolean isManifestControlFilter(String key) {
        return "pick".equals(key)
                || "index".equals(key)
                || "sort".equals(key)
                || "by".equals(key)
                || "top".equals(key)
                || "focus".equals(key)
                || "dominant".equals(key);
    }

    private static void sortManifestFiles(List<Map<String, Object>> matches, Map<String, String> filters) {
        ManifestSortSpec sortSpec = parseManifestSortSpec(filters.get("sort"));
        if (sortSpec == null) {
            return;
        }
        Comparator<Map<String, Object>> comparator = Comparator.comparing(
                file -> String.valueOf(file.getOrDefault(sortSpec.field(), "")).toLowerCase(Locale.ROOT));
        matches.sort(sortSpec.descending() ? comparator.reversed() : comparator);
    }

    private static Map<String, Object> resolveManifestFileEntry(DiffusionOpdBundleManifest manifest, String requested) {
        if (requested.contains("=")) {
            return pickManifestFile(filterManifestFilesList(manifest, requested), parseManifestFileQuery(requested));
        }
        return findManifestFileEntryByName(manifestGeneratedFileEntries(manifest), requested);
    }

    private static Map<String, Object> findManifestFileEntryByName(
            List<Map<String, Object>> entries,
            String requested) {
        for (Map<String, Object> entry : entries) {
            Object name = entry.get("name");
            if (requested.equals(String.valueOf(name))) {
                return entry;
            }
        }
        return Map.of();
    }

    private static Map<String, Object> pickManifestFile(List<Map<String, Object>> matches, ManifestFileQuery query) {
        if (matches.isEmpty()) {
            return Map.of();
        }
        if (query.index() != null) {
            if (query.index() >= 0 && query.index() < matches.size()) {
                return matches.get(query.index());
            }
            return Map.of();
        }
        return switch (query.pick()) {
            case "last" -> matches.get(matches.size() - 1);
            case "first" -> matches.get(0);
            default -> matches.get(0);
        };
    }

    record ManifestFileQuery(
            Map<String, String> filters,
            String groupBy,
            String pick,
            String sort,
            ManifestSortSpec sortSpec,
            Integer index,
            Integer top) {
    }

    record ManifestSortSpec(String field, boolean descending) {
    }
}
