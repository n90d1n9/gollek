package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Shared public CLI support for DiffusionOPD report inspection so examples and
 * JBang entrypoints can stay thin.
 *
 * <p>This class now acts mostly as the top-level inspection surface and routes
 * into focused package-private helpers for bundle export, rendering, CLI
 * parsing, manifest file queries, and manifest health analysis.
 */
public final class DiffusionOpdReportInspectorSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    static final Map<String, String> COLUMN_ALIASES = Map.ofEntries(
            Map.entry("loss", "meanLoss"),
            Map.entry("latestRound", "last.round"),
            Map.entry("latestLoss", "last.averageLoss"),
            Map.entry("latestTeacher", "last.teacherKey"),
            Map.entry("latestTask", "last.taskId"),
            Map.entry("latestStage", "last.stageName"),
            Map.entry("top1Round", "topLosses[0].round"),
            Map.entry("top1Loss", "topLosses[0].averageLoss"),
            Map.entry("top1Teacher", "topLosses[0].teacherKey"),
            Map.entry("top1Task", "topLosses[0].taskId"),
            Map.entry("top1Stage", "topLosses[0].stageName"),
            Map.entry("first1Round", "firstRounds[0].round"),
            Map.entry("first1Loss", "firstRounds[0].averageLoss"),
            Map.entry("first1Teacher", "firstRounds[0].teacherKey"),
            Map.entry("first1Task", "firstRounds[0].taskId"),
            Map.entry("first1Stage", "firstRounds[0].stageName"));

    private DiffusionOpdReportInspectorSupport() {
    }

    /**
     * Runs the shared report-inspection CLI entrypoint for either raw reports or generated bundle
     * manifests, optionally printing a caller-supplied banner first.
     */
    public static void runCli(String[] args, String... bannerLines) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the path to diffusion-opd-report.json as the first argument.");
        }
        DiffusionOpdReportInspectorCli.CliOptions options = DiffusionOpdReportInspectorCli.parseArgs(args);
        Path inputPath = Path.of(args[0]);
        printBanner(bannerLines);
        System.out.println("inputPath=" + inputPath.toAbsolutePath().normalize());
        inspectInput(inputPath, options);
    }

    private static void printBanner(String... bannerLines) {
        if (bannerLines == null || bannerLines.length == 0) {
            return;
        }
        for (String bannerLine : bannerLines) {
            System.out.println(bannerLine);
        }
    }

    private static void inspectInput(Path inputPath, DiffusionOpdReportInspectorCli.CliOptions options) {
        if (isBundleManifestInput(inputPath)) {
            inspectBundleManifest(resolveBundleManifestPath(inputPath), options.section(), options.format(), options.columns(), options.outputPath());
            return;
        }
        var report = DiffusionOpdReports.load(inputPath);
        printSection(report, inputPath, options.section(), options.format(), options.columns(), options.outputPath());
    }

    private static void printSection(
            DiffusionOpdReport report,
            Path reportPath,
            String section,
            String format,
            String columns,
            Path outputPath) {
        if (section != null && section.startsWith("bundle=")) {
            exportBundle(report, reportPath, section.substring("bundle=".length()), format, columns, outputPath);
            return;
        }
        ReportSectionView view = resolveReportSectionView(report, section, format);
        if (shouldPrintAllTextEntries(view)) {
            printNamedEntries(view.value(), columns);
            return;
        }
        printValue(view.sectionName(), view.value(), view.format(), columns, outputPath);
    }

    private static ReportSectionView resolveReportSectionView(
            DiffusionOpdReport report,
            String section,
            String format) {
        String normalizedFormat = normalizeFormat(format);
        String normalizedSection = normalizeSectionName(section);
        Object value = DiffusionOpdReports.select(report, section);
        return new ReportSectionView(normalizedSection, normalizedFormat, value);
    }

    private static boolean shouldPrintAllTextEntries(ReportSectionView view) {
        return "all".equals(view.sectionName())
                && "text".equals(view.format())
                && view.value() instanceof Map<?, ?>;
    }

    @SuppressWarnings("unchecked")
    private static void printNamedEntries(Object value, String columns) {
        Map<?, ?> map = (Map<?, ?>) value;
        map.forEach((key, element) -> printValue(String.valueOf(key), element, "text", columns, null));
    }

    private static String normalizeFormat(String format) {
        return format == null ? "text" : format.toLowerCase(Locale.ROOT);
    }

    private static String normalizeSectionName(String section) {
        return section == null ? "all" : section.toLowerCase(Locale.ROOT);
    }

    private static void printValue(String name, Object value, String format, String columns, Path outputPath) {
        emitOutput(buildOutputEmission(name, value, format, columns, outputPath));
    }

    private static OutputEmission buildOutputEmission(
            String name,
            Object value,
            String format,
            String columns,
            Path outputPath) {
        return new OutputEmission(renderValue(name, value, format, columns), outputPath);
    }

    private static void emitOutput(OutputEmission emission) {
        if (emission.outputPath() != null) {
            writeOutput(emission.outputPath(), emission.rendered());
            System.out.println("wroteOutput=" + emission.outputPath().toAbsolutePath().normalize());
            return;
        }
        System.out.print(emission.rendered());
    }

    private static void inspectBundleManifest(
            Path manifestPath,
            String section,
            String format,
            String columns,
            Path outputPath) {
        DiffusionOpdBundleView view = loadManifestView(manifestPath, section, format);
        printValue(view.section(), view.value(), view.format(), columns, outputPath);
    }

    private static DiffusionOpdBundleView loadManifestView(Path manifestPath, String section, String format) {
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.load(manifestPath);
        return inspectManifestView(manifest, section, format);
    }

    static DiffusionOpdBundleView inspectManifestView(
            DiffusionOpdBundleManifest manifest,
            String section,
            String format) {
        return DiffusionOpdReportInspectorManifestSections.inspectManifestView(manifest, section, format);
    }

    static DiffusionOpdBundleHealth inspectManifestHealth(DiffusionOpdBundleManifest manifest) {
        DiffusionOpdReportInspectorManifestHealth.ManifestHealthSnapshot snapshot =
                DiffusionOpdReportInspectorManifestHealth.collectManifestHealthSnapshot(manifest, Map.of());
        DiffusionOpdReportInspectorManifestHealth.CheckMetrics metrics =
                DiffusionOpdReportInspectorManifestHealth.collectCheckMetrics(snapshot);
        return DiffusionOpdReportInspectorManifestHealthViews.toTypedBundleHealth(manifest, snapshot, metrics);
    }

    static DiffusionOpdBundleSummary inspectManifestSummary(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorManifestSummaries.inspectManifestSummary(manifest);
    }

    static List<DiffusionOpdBundleGeneratedFile> inspectManifestFiles(DiffusionOpdBundleManifest manifest) {
        return List.copyOf(manifest.generatedFiles());
    }

    static DiffusionOpdBundleLoadedFile inspectManifestLoadedFile(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        DiffusionOpdBundleGeneratedFile file =
                DiffusionOpdReportInspectorManifestFiles.resolveTypedManifestFileEntry(manifest, requested);
        if (file == null) {
            return new DiffusionOpdBundleLoadedFile(requested, false, null, Map.of());
        }
        return new DiffusionOpdBundleLoadedFile(
                requested,
                true,
                file,
                DiffusionOpdReportInspectorManifestFiles.loadTypedBundleFileContent(manifest, file));
    }

    private static boolean isBundleManifestInput(Path inputPath) {
        return isManifestDirectoryInput(inputPath) || isManifestFileInput(inputPath);
    }

    private static boolean isManifestDirectoryInput(Path inputPath) {
        return Files.isDirectory(inputPath) && Files.exists(manifestFilePath(inputPath));
    }

    private static boolean isManifestFileInput(Path inputPath) {
        return inputPath.getFileName() != null
                && "manifest.json".equalsIgnoreCase(inputPath.getFileName().toString());
    }

    private static Path resolveBundleManifestPath(Path inputPath) {
        if (Files.isDirectory(inputPath)) {
            return resolveManifestDirectoryPath(inputPath);
        }
        return resolveManifestFilePath(inputPath);
    }

    private static Path resolveManifestDirectoryPath(Path inputPath) {
        return manifestFilePath(inputPath);
    }

    private static Path resolveManifestFilePath(Path inputPath) {
        return inputPath;
    }

    private static Path manifestFilePath(Path directoryPath) {
        return directoryPath.resolve("manifest.json");
    }

    static Object loadBundleFileContent(Path filePath, String format) {
        try {
            String normalizedFormat = format == null ? "" : format.toLowerCase(Locale.ROOT);
            if ("json".equals(normalizedFormat) || filePath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".json")) {
                return OBJECT_MAPPER.readValue(filePath.toFile(), Object.class);
            }
            return Files.readString(filePath);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load bundle file content from " + filePath + ".", exception);
        }
    }

    static String renderValue(String name, Object value, String format, String columns) {
        return DiffusionOpdReportInspectorRendering.renderByFormat(name, value, format, columns);
    }

    static String normalizeColumnsArgument(String columns) {
        if (columns == null) {
            return null;
        }
        String trimmed = columns.trim();
        return trimmed.isEmpty() || "_".equals(trimmed) ? null : trimmed;
    }

    static boolean isSupportedFormat(String value) {
        if (value == null) {
            return false;
        }
        String normalized = value.toLowerCase(Locale.ROOT);
        return "text".equals(normalized)
                || "json".equals(normalized)
                || "table".equals(normalized)
                || "csv".equals(normalized);
    }

    private record ReportSectionView(String sectionName, String format, Object value) {
    }

    private record OutputEmission(String rendered, Path outputPath) {
    }

    static String stringifyStructuredCell(Object value) {
        try {
            return OBJECT_MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException exception) {
            return String.valueOf(value);
        }
    }

    static String escapeCsv(String value) {
        boolean needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n");
        String escaped = value.replace("\"", "\"\"");
        return needsQuotes ? "\"" + escaped + "\"" : escaped;
    }

    static String renderJson(Object value) {
        try {
            return OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(value)
                    + System.lineSeparator();
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to render report output as JSON.", exception);
        }
    }

    private static void exportBundle(
            tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport report,
            Path reportPath,
            String bundleName,
            String format,
            String columns,
            Path outputPath) {
        DiffusionOpdReportInspectorBundles.exportBundle(report, reportPath, bundleName, format, columns, outputPath);
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

}
