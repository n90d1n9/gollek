package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Owns manifest section routing and render-format selection for the inspector.
 *
 * <p>This helper is the manifest-facing dispatcher: it selects the requested
 * section, then delegates file lookups, summary shaping, and health shaping to
 * the narrower manifest helpers.
 */
final class DiffusionOpdReportInspectorManifestSections {

    private DiffusionOpdReportInspectorManifestSections() {
    }

    static DiffusionOpdBundleView inspectManifestView(
            DiffusionOpdBundleManifest manifest,
            String section,
            String format) {
        String normalizedSection = (section == null ? "manifest" : section).toLowerCase(Locale.ROOT);
        Object value = selectManifestSection(manifest, normalizedSection);
        return new DiffusionOpdBundleView(
                normalizedSection,
                effectiveManifestRenderFormat(normalizedSection, value, format),
                value);
    }

    private static Object selectManifestSection(DiffusionOpdBundleManifest manifest, String section) {
        return resolveManifestSelection(manifest, parseManifestSelection(section));
    }

    private static Object resolveManifestSelection(
            DiffusionOpdBundleManifest manifest,
            ManifestSelection selection) {
        return switch (selection.command()) {
            case "manifest", "all", "overview" -> manifest.toMap();
            case "bundlesummary", "manifestsummary" ->
                    DiffusionOpdReportInspectorManifestSummaries.summarizeManifestBundle(manifest, selection.argument());
            case "bundlehealth", "manifesthealth" -> summarizeManifestHealth(manifest, selection.argument());
            case "files", "generatedfiles" -> selection.argument().isBlank()
                    ? DiffusionOpdReportInspectorManifestFiles.manifestGeneratedFileEntries(manifest)
                    : DiffusionOpdReportInspectorManifestFiles.filterManifestFilesList(manifest, selection.argument());
            case "filessummary" ->
                    DiffusionOpdReportInspectorManifestSummaries.summarizeManifestFiles(manifest, selection.argument());
            case "loadfile" -> DiffusionOpdReportInspectorManifestFiles.loadManifestFileContent(manifest, selection.argument());
            case "file" -> DiffusionOpdReportInspectorManifestFiles.findManifestFile(manifest, selection.argument());
            default -> manifest.toMap().getOrDefault(selection.fallbackSection(), manifest.toMap());
        };
    }

    private static ManifestSelection parseManifestSelection(String section) {
        return buildManifestSelection(section, section.indexOf(':'));
    }

    private static ManifestSelection buildManifestSelection(String section, int delimiter) {
        if (delimiter < 0) {
            return new ManifestSelection(section, "", section);
        }
        return new ManifestSelection(
                section.substring(0, delimiter),
                section.substring(delimiter + 1),
                section);
    }

    private static String effectiveManifestRenderFormat(String section, Object value, String format) {
        String normalizedFormat = normalizeManifestRenderFormat(format);
        if (shouldPromoteManifestLoadfileToJson(section, value, normalizedFormat)) {
            return "json";
        }
        return normalizedFormat;
    }

    private static String normalizeManifestRenderFormat(String format) {
        return format == null ? "text" : format.toLowerCase(Locale.ROOT);
    }

    private static boolean shouldPromoteManifestLoadfileToJson(
            String section,
            Object value,
            String normalizedFormat) {
        return section.startsWith("loadfile:")
                && "text".equals(normalizedFormat)
                && (value instanceof Map<?, ?> || value instanceof List<?>);
    }

    private static Map<String, Object> summarizeManifestHealth(
            DiffusionOpdBundleManifest manifest,
            String optionsExpression) {
        Map<String, String> options = DiffusionOpdReportInspectorManifestFiles.parseManifestFilters(optionsExpression);
        DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot =
                DiffusionOpdReportInspectorManifestHealth.collectManifestHealthSnapshot(manifest, options);
        String focus = options.getOrDefault("focus", "all").trim().toLowerCase(Locale.ROOT);
        LinkedHashMap<String, Object> health = new LinkedHashMap<>();
        health.put("bundleType", manifest.bundleType());
        DiffusionOpdReportInspectorManifestHealthViews.applyManifestHealthRoot(health, snapshot);
        DiffusionOpdReportInspectorManifestHealthViews.applyManifestHealthChecks(health, snapshot);
        applyManifestHealthFocus(health, snapshot, focus, options);
        return Map.copyOf(health);
    }

    private static void applyManifestHealthFocus(
            LinkedHashMap<String, Object> health,
            DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot,
            String focus,
            Map<String, String> options) {
        if (!snapshot.missingFiles().isEmpty() && ("all".equals(focus) || "files".equals(focus))) {
            health.put("missingFiles", DiffusionOpdReportInspectorManifestSummaries.limitMissingFiles(snapshot.missingFiles(), options));
        }
        if (!snapshot.missingFiles().isEmpty() && ("all".equals(focus) || "sections".equals(focus))) {
            health.put("missingSections", snapshot.missingSections());
        }
        if (!snapshot.missingFiles().isEmpty() && ("all".equals(focus) || "formats".equals(focus))) {
            health.put("missingFormats", snapshot.missingFormats());
        }
        health.put("focus", focus);
        if (options.containsKey("top")) {
            health.put("top", options.get("top"));
        }
    }

    private record ManifestSelection(String command, String argument, String fallbackSection) {
    }
}
