package tech.kayys.gollek.train.diffusion.opd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Owns bundle-export planning and writing for the inspector CLI.
 *
 * <p>This helper stays at the artifact-export boundary while
 * {@link DiffusionOpdReportInspectorRendering} handles section rendering and
 * {@link DiffusionOpdReportInspectorManifestSections} handles manifest-specific routing.
 */
final class DiffusionOpdReportInspectorBundles {

    private DiffusionOpdReportInspectorBundles() {
    }

    static void exportBundle(
            DiffusionOpdReport report,
            Path reportPath,
            String bundleName,
            String format,
            String columns,
            Path outputPath) {
        if (outputPath == null) {
            throw new IllegalArgumentException("Bundle export requires an output directory path.");
        }
        Map<String, BundleEntry> bundleSections = bundleSections(bundleName);
        Path bundleDir = outputPath.toAbsolutePath().normalize();
        ensureBundleDirectory(bundleDir);
        List<BundleArtifact> artifacts = planBundleArtifacts(report, bundleDir, bundleSections, format, columns);
        writeBundleArtifacts(artifacts);
        writeBundleManifest(bundleDir, bundleName, reportPath, format, columns, artifacts);
        System.out.println("wroteBundle=" + bundleDir);
    }

    private static void ensureBundleDirectory(Path bundleDir) {
        try {
            Files.createDirectories(bundleDir);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to create bundle output directory " + bundleDir + ".", exception);
        }
    }

    private static List<BundleArtifact> planBundleArtifacts(
            DiffusionOpdReport report,
            Path bundleDir,
            Map<String, BundleEntry> bundleSections,
            String format,
            String columns) {
        List<BundleArtifact> artifacts = new ArrayList<>();
        for (Map.Entry<String, BundleEntry> entry : bundleSections.entrySet()) {
            artifacts.add(planBundleArtifact(report, bundleDir, entry.getKey(), entry.getValue(), format, columns));
        }
        return List.copyOf(artifacts);
    }

    private static BundleArtifact planBundleArtifact(
            DiffusionOpdReport report,
            Path bundleDir,
            String outputName,
            BundleEntry entry,
            String format,
            String columns) {
        String effectiveFormat = bundleFormatForEntry(format, entry.format());
        Object value = DiffusionOpdReports.select(report, entry.section());
        String effectiveColumns = bundleColumnsForSection(
                entry.section(),
                effectiveFormat,
                columns,
                entry.columns());
        String rendered = DiffusionOpdReportInspectorSupport.renderValue(
                outputName,
                value,
                effectiveFormat,
                effectiveColumns);
        Path file = bundleDir.resolve(outputName + formatExtension(effectiveFormat));
        return new BundleArtifact(
                file,
                entry.section(),
                effectiveFormat,
                entry.format(),
                effectiveColumns,
                entry.columns(),
                rendered);
    }

    private static void writeBundleArtifacts(List<BundleArtifact> artifacts) {
        for (BundleArtifact artifact : artifacts) {
            writeOutput(artifact.file(), artifact.rendered());
        }
    }

    private static void writeBundleManifest(
            Path bundleDir,
            String bundleName,
            Path reportPath,
            String format,
            String columns,
            List<BundleArtifact> artifacts) {
        writeOutput(bundleDir.resolve("manifest.json"), DiffusionOpdReportInspectorSupport.renderJson(Map.of(
                "bundle", bundleName,
                "createdAt", Instant.now().toString(),
                "format", format,
                "columns", columns,
                "columnStrategy", columns == null ? "bundle-defaults" : "explicit",
                "sourceReportPath", reportPath.toAbsolutePath().normalize().toString(),
                "outputDirectory", bundleDir.toString(),
                "generatedFiles", manifestGeneratedFiles(artifacts))));
    }

    private static List<Map<String, Object>> manifestGeneratedFiles(List<BundleArtifact> artifacts) {
        List<Map<String, Object>> generatedFiles = new ArrayList<>();
        for (BundleArtifact artifact : artifacts) {
            generatedFiles.add(Map.of(
                    "name", artifact.file().getFileName().toString(),
                    "section", artifact.section(),
                    "format", artifact.format(),
                    "entryFormat", artifact.entryFormat(),
                    "columns", artifact.columns(),
                    "entryColumns", artifact.entryColumns()));
        }
        generatedFiles.add(Map.of(
                "name", "manifest.json",
                "section", "bundle-manifest",
                "format", "json"));
        return List.copyOf(generatedFiles);
    }

    private static Map<String, BundleEntry> bundleSections(String bundleName) {
        String normalized = bundleName.toLowerCase(Locale.ROOT);
        if (normalized.startsWith("custom:")) {
            return customBundleSections(bundleName.substring("custom:".length()));
        }
        return builtInBundleSections(bundleName, normalized);
    }

    private static Map<String, BundleEntry> builtInBundleSections(String bundleName, String normalized) {
        return switch (normalized) {
            case "standard" -> Map.of(
                    "overview", new BundleEntry("overview", null, null),
                    "tasks", new BundleEntry("taskSummaries", null, null),
                    "teachers", new BundleEntry("teacherSummaries", null, null),
                    "stages", new BundleEntry("stageSummaries", null, null));
            case "rollups" -> Map.of(
                    "tasks", new BundleEntry("taskSummaries", null, null),
                    "teachers", new BundleEntry("teacherSummaries", null, null),
                    "stages", new BundleEntry("stageSummaries", null, null),
                    "taskTeachers", new BundleEntry("taskTeacherSummaries", null, null),
                    "taskStages", new BundleEntry("taskStageSummaries", null, null),
                    "teacherStages", new BundleEntry("teacherStageSummaries", null, null));
            default -> throw new IllegalArgumentException(
                    "Unknown bundle '" + bundleName
                            + "'. Use bundle=standard, bundle=rollups, or bundle=custom:section1,section2.");
        };
    }

    private static Map<String, BundleEntry> customBundleSections(String definition) {
        LinkedHashMap<String, BundleEntry> sections = new LinkedHashMap<>();
        for (String token : definition.split(",")) {
            String normalizedToken = token.trim();
            if (normalizedToken.isEmpty()) {
                continue;
            }
            addCustomBundleSection(sections, normalizedToken);
        }
        if (sections.isEmpty()) {
            throw new IllegalArgumentException(
                    "Custom bundles must include at least one section, for example bundle=custom:overview,taskSummaries,"
                            + " bundle=custom:overview@run-overview,"
                            + " bundle=custom:taskSummaries@tasks#compact,"
                            + " or bundle=custom:overview@run-overview!json.");
        }
        return Map.copyOf(sections);
    }

    private static void addCustomBundleSection(Map<String, BundleEntry> sections, String token) {
        CustomBundleToken parsed = parseCustomBundleToken(token);
        String section = parsed.section();
        if (section.isEmpty()) {
            return;
        }
        String outputName = uniqueBundleAlias(
                sections,
                sanitizeBundleAlias(customBundleAlias(parsed, section)));
        sections.put(outputName, toBundleEntry(parsed, section));
    }

    private static String customBundleAlias(CustomBundleToken token, String fallbackSection) {
        return token.alias() == null || token.alias().isEmpty() ? fallbackSection : token.alias();
    }

    private static BundleEntry toBundleEntry(CustomBundleToken token, String section) {
        return new BundleEntry(
                section,
                DiffusionOpdReportInspectorSupport.normalizeColumnsArgument(token.columns()),
                normalizeEntryFormat(token.format()));
    }

    private static CustomBundleToken parseCustomBundleToken(String token) {
        BundleTokenMarkers markers = detectBundleTokenMarkers(token);
        return new CustomBundleToken(
                parseCustomBundleSection(token, markers),
                parseCustomBundleField(token, markers.aliasMarker(), markers),
                parseCustomBundleField(token, markers.columnsMarker(), markers),
                parseCustomBundleField(token, markers.formatMarker(), markers));
    }

    private static BundleTokenMarkers detectBundleTokenMarkers(String token) {
        return new BundleTokenMarkers(
                token.lastIndexOf('@'),
                token.lastIndexOf('#'),
                token.lastIndexOf('!'));
    }

    private static String parseCustomBundleSection(String token, BundleTokenMarkers markers) {
        int firstSplit = firstBundleTokenSplit(token.length(), markers);
        return firstSplit < token.length() ? token.substring(0, firstSplit).trim() : token;
    }

    private static int firstBundleTokenSplit(int tokenLength, BundleTokenMarkers markers) {
        int firstSplit = tokenLength;
        if (markers.aliasMarker() >= 0) {
            firstSplit = Math.min(firstSplit, markers.aliasMarker());
        }
        if (markers.columnsMarker() >= 0) {
            firstSplit = Math.min(firstSplit, markers.columnsMarker());
        }
        if (markers.formatMarker() >= 0) {
            firstSplit = Math.min(firstSplit, markers.formatMarker());
        }
        return firstSplit;
    }

    private static String parseCustomBundleField(String token, int marker, BundleTokenMarkers markers) {
        if (marker < 0) {
            return null;
        }
        return token.substring(marker + 1, customBundleFieldEnd(token.length(), marker, markers)).trim();
    }

    private static int customBundleFieldEnd(int tokenLength, int marker, BundleTokenMarkers markers) {
        int fieldEnd = tokenLength;
        if (markers.aliasMarker() > marker) {
            fieldEnd = Math.min(fieldEnd, markers.aliasMarker());
        }
        if (markers.columnsMarker() > marker) {
            fieldEnd = Math.min(fieldEnd, markers.columnsMarker());
        }
        if (markers.formatMarker() > marker) {
            fieldEnd = Math.min(fieldEnd, markers.formatMarker());
        }
        return fieldEnd;
    }

    private static String sanitizeBundleAlias(String section) {
        String alias = section.replaceAll("[^A-Za-z0-9._-]+", "_");
        if (alias.isEmpty()) {
            return "section";
        }
        return alias;
    }

    private static String uniqueBundleAlias(Map<String, BundleEntry> sections, String alias) {
        if (!sections.containsKey(alias)) {
            return alias;
        }
        int suffix = 2;
        while (sections.containsKey(alias + "_" + suffix)) {
            suffix++;
        }
        return alias + "_" + suffix;
    }

    private static String formatExtension(String format) {
        return switch (format) {
            case "json" -> ".json";
            case "csv" -> ".csv";
            case "table" -> ".table";
            case "text" -> ".txt";
            default -> throw new IllegalArgumentException(
                    "Unsupported bundle format '" + format + "'. Use text, json, table, or csv.");
        };
    }

    private static String normalizeEntryFormat(String format) {
        if (format == null || format.isBlank()) {
            return null;
        }
        String normalized = format.toLowerCase(Locale.ROOT);
        if (!DiffusionOpdReportInspectorSupport.isSupportedFormat(normalized)) {
            throw new IllegalArgumentException(
                    "Unsupported entry format '" + format + "'. Use text, json, table, or csv.");
        }
        return normalized;
    }

    private static String bundleFormatForEntry(String explicitFormat, String entryFormat) {
        if (explicitFormat != null) {
            return explicitFormat;
        }
        if (entryFormat != null) {
            return entryFormat;
        }
        return "text";
    }

    private static String bundleColumnsForSection(
            String section,
            String format,
            String explicitColumns,
            String entryColumns) {
        if (explicitColumns != null) {
            return explicitColumns;
        }
        if (entryColumns != null) {
            return entryColumns;
        }
        if (!"csv".equals(format) && !"table".equals(format)) {
            return null;
        }
        return switch (section) {
            case "taskSummaries", "teacherSummaries", "stageSummaries" -> "compact";
            case "taskTeacherSummaries", "taskStageSummaries", "teacherStageSummaries" -> "compare";
            default -> null;
        };
    }

    private static void writeOutput(Path outputPath, String content) {
        try {
            Path absolute = outputPath.toAbsolutePath().normalize();
            Path parent = absolute.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(absolute, content);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write inspector output to " + outputPath + ".", exception);
        }
    }

    private record BundleArtifact(
            Path file,
            String section,
            String format,
            String entryFormat,
            String columns,
            String entryColumns,
            String rendered) {
    }

    private record BundleTokenMarkers(int aliasMarker, int columnsMarker, int formatMarker) {
    }

    private record BundleEntry(String section, String columns, String format) {
    }

    private record CustomBundleToken(String section, String alias, String columns, String format) {
    }
}
