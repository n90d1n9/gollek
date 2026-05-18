package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Shared CLI support for DiffusionOPD report inspection so examples and JBang
 * entrypoints can stay thin.
 */
public final class DiffusionOpdReportInspectorSupport {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final Map<String, String> COLUMN_ALIASES = Map.ofEntries(
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

    public static void runCli(String[] args, String... bannerLines) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the path to diffusion-opd-report.json as the first argument.");
        }
        CliOptions options = parseArgs(args);
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

    private static void inspectInput(Path inputPath, CliOptions options) {
        if (isBundleManifestInput(inputPath)) {
            inspectBundleManifest(resolveBundleManifestPath(inputPath), options.section(), options.format(), options.columns(), options.outputPath());
            return;
        }
        var report = DiffusionOpdReports.load(inputPath);
        printSection(report, inputPath, options.section(), options.format(), options.columns(), options.outputPath());
    }

    private static void printSection(
            tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport report,
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
            tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport report,
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
                && view.value() instanceof java.util.Map<?, ?>;
    }

    @SuppressWarnings("unchecked")
    private static void printNamedEntries(Object value, String columns) {
        java.util.Map<?, ?> map = (java.util.Map<?, ?>) value;
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
        DiffusionOpdBundleManifest manifest = loadBundleManifest(manifestPath);
        return inspectManifestView(manifest, section, format);
    }

    static DiffusionOpdBundleView inspectManifestView(
            DiffusionOpdBundleManifest manifest,
            String section,
            String format) {
        String normalizedSection = normalizeManifestSection(section);
        Object value = selectManifestSection(manifest, normalizedSection);
        return new DiffusionOpdBundleView(
                normalizedSection,
                effectiveManifestRenderFormat(normalizedSection, value, format),
                value);
    }

    static DiffusionOpdBundleHealth inspectManifestHealth(DiffusionOpdBundleManifest manifest) {
        ManifestHealthSnapshot snapshot = collectManifestHealthSnapshot(manifest, Map.of());
        CheckMetrics metrics = collectCheckMetrics(snapshot);
        return toTypedBundleHealth(manifest, snapshot, metrics);
    }

    static DiffusionOpdBundleSummary inspectManifestSummary(DiffusionOpdBundleManifest manifest) {
        return toTypedBundleSummary(manifest, Map.of());
    }

    static List<DiffusionOpdBundleGeneratedFile> inspectManifestFiles(DiffusionOpdBundleManifest manifest) {
        return List.copyOf(manifest.generatedFiles());
    }

    static DiffusionOpdBundleLoadedFile inspectManifestLoadedFile(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        DiffusionOpdBundleGeneratedFile file = resolveTypedManifestFileEntry(manifest, requested);
        if (file == null) {
            return new DiffusionOpdBundleLoadedFile(requested, false, null, Map.of());
        }
        return new DiffusionOpdBundleLoadedFile(
                requested,
                true,
                file,
                loadTypedBundleFileContent(manifest, file));
    }

    private static String normalizeManifestSection(String section) {
        return normalizeManifestSectionToken(defaultManifestSection(section));
    }

    private static String defaultManifestSection(String section) {
        return section == null ? "manifest" : section;
    }

    private static String normalizeManifestSectionToken(String section) {
        return section.toLowerCase(Locale.ROOT);
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

    private static DiffusionOpdBundleManifest loadBundleManifest(Path manifestPath) {
        return DiffusionOpdBundleManifests.load(manifestPath);
    }

    private static Object selectManifestSection(DiffusionOpdBundleManifest manifest, String section) {
        return resolveManifestSelection(manifest, parseManifestSelection(section));
    }

    private static Object resolveManifestSelection(DiffusionOpdBundleManifest manifest, ManifestSelection selection) {
        return switch (selection.command()) {
            case "manifest", "all", "overview" -> manifest.toMap();
            case "bundlesummary", "manifestsummary" -> summarizeManifestBundle(manifest, selection.argument());
            case "bundlehealth", "manifesthealth" -> summarizeManifestHealth(manifest, selection.argument());
            case "files", "generatedfiles" -> selection.argument().isBlank()
                    ? manifestGeneratedFileEntries(manifest)
                    : filterManifestFiles(manifest, selection.argument());
            case "filessummary" -> summarizeManifestFiles(manifest, selection.argument());
            case "loadfile" -> loadManifestFileContent(manifest, selection.argument());
            case "file" -> findManifestFile(manifest, selection.argument());
            default -> manifest.toMap().getOrDefault(selection.fallbackSection(), manifest.toMap());
        };
    }

    private static ManifestSelection parseManifestSelection(String section) {
        return buildManifestSelection(section, manifestSelectionDelimiter(section));
    }

    private static int manifestSelectionDelimiter(String section) {
        return section.indexOf(':');
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

    private static Object filterManifestFiles(DiffusionOpdBundleManifest manifest, String filterExpression) {
        return filterManifestFilesList(manifest, filterExpression);
    }

    private static Map<String, Object> summarizeManifestHealth(DiffusionOpdBundleManifest manifest) {
        return summarizeManifestHealth(manifest, "");
    }

    private static Map<String, Object> summarizeManifestHealth(DiffusionOpdBundleManifest manifest, String optionsExpression) {
        Map<String, String> options = parseManifestFilters(optionsExpression);
        ManifestHealthSnapshot snapshot = collectManifestHealthSnapshot(manifest, options);
        String focus = options.getOrDefault("focus", "all").trim().toLowerCase(Locale.ROOT);
        LinkedHashMap<String, Object> health = new LinkedHashMap<>();
        health.put("bundleType", manifest.bundleType());
        applyManifestHealthRoot(health, snapshot);
        applyManifestHealthChecks(health, snapshot);
        applyManifestHealthFocus(health, snapshot, focus, options);
        return Map.copyOf(health);
    }

    private static void applyManifestHealthFocus(
            LinkedHashMap<String, Object> health,
            ManifestHealthSnapshot snapshot,
            String focus,
            Map<String, String> options) {
        if (!snapshot.missingFiles().isEmpty() && ("all".equals(focus) || "files".equals(focus))) {
            health.put("missingFiles", limitMissingFiles(snapshot.missingFiles(), options));
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

    private static DiffusionOpdBundleHealth toTypedBundleHealth(
            DiffusionOpdBundleManifest manifest,
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics) {
        return new DiffusionOpdBundleHealth(
                manifest.bundleType(),
                snapshot.outputDirectory(),
                snapshot.outputDirectoryConfigured(),
                snapshot.outputDirectoryExists(),
                snapshot.totalFiles(),
                snapshot.existingFileCount(),
                snapshot.missingFileCount(),
                snapshot.healthy(),
                snapshot.healthStatus(),
                snapshot.healthScore(),
                snapshot.alertLevel(),
                snapshot.issueCodes(),
                snapshot.primaryIssueCode(),
                toTypedBundleHealthBadge(snapshot, metrics),
                snapshot.summaryMessage(),
                snapshot.recommendedAction(),
                snapshot.checks().stream()
                        .map(DiffusionOpdReportInspectorSupport::toTypedBundleHealthCheck)
                        .toList(),
                snapshot.failingChecks().stream()
                        .map(DiffusionOpdReportInspectorSupport::toTypedBundleHealthCheck)
                        .toList(),
                snapshot.failingChecks().size(),
                metrics.passingCheckCount(),
                metrics.criticalCheckCount(),
                metrics.warningCheckCount(),
                metrics.infoCheckCount(),
                toTypedBundleCheckSummary(snapshot, metrics),
                snapshot.missingFiles(),
                snapshot.missingSections().stream()
                        .map(DiffusionOpdReportInspectorSupport::toTypedGroupedCount)
                        .toList(),
                snapshot.missingFormats().stream()
                        .map(DiffusionOpdReportInspectorSupport::toTypedGroupedCount)
                        .toList());
    }

    private static DiffusionOpdBundleSummary toTypedBundleSummary(
            DiffusionOpdBundleManifest manifest,
            Map<String, String> options) {
        List<Map<String, Object>> files = filterManifestFilesList(manifest, "");
        List<Map<String, Object>> bySection = limitManifestSummaryRows(
                castSummaryRows(summarizeManifestFiles(manifest, "by=section:sort=-count")),
                options);
        List<Map<String, Object>> byFormat = limitManifestSummaryRows(
                castSummaryRows(summarizeManifestFiles(manifest, "by=format:sort=-count")),
                options);
        String focus = options.getOrDefault("focus", "all").trim().toLowerCase(Locale.ROOT);
        boolean dominantOnly = Boolean.parseBoolean(options.getOrDefault("dominant", "false"));
        List<String> missingFiles = findMissingManifestFiles(manifest, files);
        List<DiffusionOpdBundleGroupedCount> typedSections = bySection.stream()
                .map(DiffusionOpdReportInspectorSupport::toTypedGroupedCount)
                .toList();
        List<DiffusionOpdBundleGroupedCount> typedFormats = byFormat.stream()
                .map(DiffusionOpdReportInspectorSupport::toTypedGroupedCount)
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
                missingFiles);
    }

    private static List<String> limitMissingFiles(List<String> missingFiles, Map<String, String> options) {
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

    private static ManifestHealthSnapshot collectManifestHealthSnapshot(
            DiffusionOpdBundleManifest manifest,
            Map<String, String> options) {
        List<Map<String, Object>> files = filterManifestFilesList(manifest, "");
        List<Map<String, Object>> missingEntries = findMissingManifestFileEntries(manifest, files);
        List<String> missingFiles = missingEntries.stream()
                .map(entry -> String.valueOf(entry.getOrDefault("name", "")))
                .toList();
        String outputDirectory = manifest.outputDirectory();
        boolean outputDirectoryConfigured = !outputDirectory.isBlank();
        boolean outputDirectoryExists = outputDirectoryConfigured && Files.isDirectory(Path.of(outputDirectory));
        int totalFiles = files.size();
        int missingFileCount = missingFiles.size();
        int existingFileCount = totalFiles - missingFileCount;
        boolean healthy = missingFiles.isEmpty();
        double healthScore = totalFiles == 0 ? 1.0d : Math.max(0.0d, (double) existingFileCount / (double) totalFiles);
        String healthStatus = classifyBundleHealth(totalFiles, missingFileCount);
        String summaryMessage = buildBundleHealthSummary(healthStatus, totalFiles, missingFileCount);
        String recommendedAction = recommendedBundleHealthAction(healthStatus);
        List<String> issueCodes = bundleHealthIssueCodes(healthStatus, missingFileCount);
        String primaryIssueCode = issueCodes.isEmpty() ? "bundle.health.unknown" : issueCodes.get(0);
        String alertLevel = bundleHealthAlertLevel(healthStatus);
        List<Map<String, Object>> checks = buildBundleHealthChecks(
                outputDirectoryConfigured,
                outputDirectoryExists,
                totalFiles,
                missingFileCount,
                healthStatus);
        List<Map<String, Object>> failingChecks = checks.stream()
                .filter(check -> !Boolean.TRUE.equals(check.get("passed")))
                .toList();
        List<Map<String, Object>> missingSections = limitManifestSummaryRows(
                summarizeMissingEntriesByField(missingEntries, "section"),
                options);
        List<Map<String, Object>> missingFormats = limitManifestSummaryRows(
                summarizeMissingEntriesByField(missingEntries, "format"),
                options);
        return new ManifestHealthSnapshot(
                outputDirectory,
                outputDirectoryConfigured,
                outputDirectoryExists,
                totalFiles,
                existingFileCount,
                missingFileCount,
                healthy,
                healthScore,
                healthStatus,
                alertLevel,
                issueCodes,
                primaryIssueCode,
                summaryMessage,
                recommendedAction,
                checks,
                failingChecks,
                missingFiles,
                missingSections,
                missingFormats);
    }

    private static void applyManifestHealthRoot(
            LinkedHashMap<String, Object> health,
            ManifestHealthSnapshot snapshot) {
        health.put("outputDirectory", snapshot.outputDirectory());
        health.put("outputDirectoryConfigured", snapshot.outputDirectoryConfigured());
        health.put("outputDirectoryExists", snapshot.outputDirectoryExists());
        health.put("totalFiles", snapshot.totalFiles());
        health.put("existingFileCount", snapshot.existingFileCount());
        health.put("missingFileCount", snapshot.missingFileCount());
        health.put("healthy", snapshot.healthy());
        health.put("status", snapshot.healthStatus());
        health.put("healthScore", snapshot.healthScore());
        health.put("alertLevel", snapshot.alertLevel());
        health.put("issueCodes", snapshot.issueCodes());
        health.put("primaryIssueCode", snapshot.primaryIssueCode());
        health.put("healthBadge", buildBundleHealthBadge(snapshot));
        health.put("summaryMessage", snapshot.summaryMessage());
        health.put("recommendedAction", snapshot.recommendedAction());
        health.put("checks", snapshot.checks());
    }

    private static void applyManifestHealthChecks(
            LinkedHashMap<String, Object> health,
            ManifestHealthSnapshot snapshot) {
        CheckMetrics metrics = collectCheckMetrics(snapshot);
        health.put("healthBadge", buildBundleHealthBadge(snapshot, metrics));
        health.put("failingChecks", snapshot.failingChecks());
        health.put("failingCheckCount", snapshot.failingChecks().size());
        health.put("passingCheckCount", metrics.passingCheckCount());
        health.put("criticalCheckCount", metrics.criticalCheckCount());
        health.put("warningCheckCount", metrics.warningCheckCount());
        health.put("infoCheckCount", metrics.infoCheckCount());
        health.put("checkSummary", buildCheckSummary(snapshot, metrics));
    }

    private static Map<String, Object> buildBundleHealthBadge(ManifestHealthSnapshot snapshot) {
        return Map.of(
                "status", snapshot.healthStatus(),
                "alertLevel", snapshot.alertLevel(),
                "primaryIssueCode", snapshot.primaryIssueCode(),
                "score", snapshot.healthScore(),
                "variant", bundleHealthBadgeVariant(snapshot.healthStatus()),
                "label", bundleHealthBadgeLabel(snapshot.healthStatus()),
                "token", bundleHealthBadgeToken(snapshot.healthStatus()),
                "tooltip", snapshot.summaryMessage(),
                "checkStatus", "pending");
    }

    private static DiffusionOpdBundleHealthBadge toTypedBundleHealthBadge(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics) {
        return new DiffusionOpdBundleHealthBadge(
                snapshot.healthStatus(),
                snapshot.alertLevel(),
                snapshot.primaryIssueCode(),
                metrics.primaryFailingCheckCode(),
                metrics.primaryFailingCheckSeverity(),
                snapshot.healthScore(),
                bundleHealthBadgeVariant(snapshot.healthStatus()),
                bundleHealthBadgeLabel(snapshot.healthStatus()),
                bundleHealthBadgeToken(snapshot.healthStatus()),
                snapshot.summaryMessage(),
                snapshot.failingChecks().isEmpty() ? "pass" : "fail");
    }

    private static Map<String, Object> buildBundleHealthBadge(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics) {
        LinkedHashMap<String, Object> badge = new LinkedHashMap<>();
        badge.put("status", snapshot.healthStatus());
        badge.put("alertLevel", snapshot.alertLevel());
        badge.put("primaryIssueCode", snapshot.primaryIssueCode());
        badge.put("primaryCheckCode", metrics.primaryFailingCheckCode());
        badge.put("primaryCheckSeverity", metrics.primaryFailingCheckSeverity());
        badge.put("score", snapshot.healthScore());
        badge.put("variant", bundleHealthBadgeVariant(snapshot.healthStatus()));
        badge.put("label", bundleHealthBadgeLabel(snapshot.healthStatus()));
        badge.put("token", bundleHealthBadgeToken(snapshot.healthStatus()));
        badge.put("tooltip", snapshot.summaryMessage());
        badge.put("checkStatus", snapshot.failingChecks().isEmpty() ? "pass" : "fail");
        return badge;
    }

    private static Map<String, Object> buildCheckSummary(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics) {
        String checkStatus = snapshot.failingChecks().isEmpty() ? "healthy" : snapshot.healthStatus();
        String checkAlertLevel = snapshot.failingChecks().isEmpty() ? "info" : bundleHealthAlertLevel(snapshot.healthStatus());
        String checkSummaryMessage = snapshot.failingChecks().isEmpty()
                ? "All bundle health checks passed."
                : "Bundle health checks reported " + snapshot.failingChecks().size() + " failure(s).";
        String checkRecommendedAction = snapshot.failingChecks().isEmpty()
                ? "none"
                : ("critical".equalsIgnoreCase(metrics.dominantSeverity())
                        ? "repair-critical-checks"
                        : "review-failing-checks");
        List<String> checkIssueCodes = snapshot.failingChecks().isEmpty()
                ? List.of("checks.ok")
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .distinct()
                        .toList();
        String primaryCheckIssueCode = snapshot.failingChecks().isEmpty()
                ? "checks.ok"
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .findFirst()
                        .orElse("checks.unknown");
        Map<String, Object> nestedBadge = buildCheckSummaryBadge(
                snapshot,
                metrics,
                checkStatus,
                checkAlertLevel,
                primaryCheckIssueCode,
                checkSummaryMessage,
                checkRecommendedAction);
        LinkedHashMap<String, Object> summary = new LinkedHashMap<>();
        summary.put("total", snapshot.checks().size());
        summary.put("passed", metrics.passingCheckCount());
        summary.put("failed", snapshot.failingChecks().size());
        summary.put("passRate", snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size());
        summary.put("failureRate", snapshot.checks().isEmpty() ? 0.0d : (double) snapshot.failingChecks().size() / (double) snapshot.checks().size());
        summary.put("status", checkStatus);
        summary.put("alertLevel", checkAlertLevel);
        summary.put("summaryMessage", checkSummaryMessage);
        summary.put("recommendedAction", checkRecommendedAction);
        summary.put("issueCodes", checkIssueCodes);
        summary.put("primaryIssueCode", primaryCheckIssueCode);
        summary.put("healthBadge", nestedBadge);
        summary.put("dominantSeverity", metrics.dominantSeverity());
        summary.put("allPassed", snapshot.failingChecks().isEmpty());
        summary.put("hasCriticalFailures", hasFailingChecksBySeverity(snapshot.failingChecks(), "critical"));
        summary.put("hasWarningFailures", hasFailingChecksBySeverity(snapshot.failingChecks(), "warning"));
        summary.put("hasInfoFailures", hasFailingChecksBySeverity(snapshot.failingChecks(), "info"));
        summary.put("failingSeverityCounts", Map.of(
                "critical", metrics.failingCriticalCheckCount(),
                "warning", metrics.failingWarningCheckCount(),
                "info", metrics.failingInfoCheckCount()));
        summary.put("primaryFailingName", metrics.primaryFailingCheckName());
        summary.put("primaryFailingCode", metrics.primaryFailingCheckCode());
        summary.put("primaryFailingSeverity", metrics.primaryFailingCheckSeverity());
        summary.put("primaryFailingMessage", metrics.primaryFailingCheckMessage());
        summary.put("failingNames", snapshot.failingChecks().stream()
                .map(check -> String.valueOf(check.getOrDefault("name", "")))
                .toList());
        summary.put("failingCodes", snapshot.failingChecks().stream()
                .map(check -> String.valueOf(check.getOrDefault("code", "")))
                .toList());
        summary.put("severityCounts", Map.of(
                "critical", metrics.criticalCheckCount(),
                "warning", metrics.warningCheckCount(),
                "info", metrics.infoCheckCount()));
        return summary;
    }

    private static DiffusionOpdBundleCheckSummary toTypedBundleCheckSummary(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics) {
        String checkStatus = snapshot.failingChecks().isEmpty() ? "healthy" : snapshot.healthStatus();
        String checkAlertLevel = snapshot.failingChecks().isEmpty()
                ? "info"
                : bundleHealthAlertLevel(snapshot.healthStatus());
        String checkSummaryMessage = snapshot.failingChecks().isEmpty()
                ? "All bundle health checks passed."
                : "Bundle health checks reported " + snapshot.failingChecks().size() + " failure(s).";
        String checkRecommendedAction = snapshot.failingChecks().isEmpty()
                ? "none"
                : ("critical".equalsIgnoreCase(metrics.dominantSeverity())
                        ? "repair-critical-checks"
                        : "review-failing-checks");
        List<String> checkIssueCodes = snapshot.failingChecks().isEmpty()
                ? List.of("checks.ok")
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .distinct()
                        .toList();
        String primaryCheckIssueCode = snapshot.failingChecks().isEmpty()
                ? "checks.ok"
                : snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .filter(code -> !code.isBlank())
                        .findFirst()
                        .orElse("checks.unknown");
        return new DiffusionOpdBundleCheckSummary(
                snapshot.checks().size(),
                metrics.passingCheckCount(),
                snapshot.failingChecks().size(),
                snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size(),
                snapshot.checks().isEmpty() ? 0.0d : (double) snapshot.failingChecks().size() / (double) snapshot.checks().size(),
                checkStatus,
                checkAlertLevel,
                checkSummaryMessage,
                checkRecommendedAction,
                checkIssueCodes,
                primaryCheckIssueCode,
                toTypedBundleCheckSummaryBadge(
                        snapshot,
                        metrics,
                        checkStatus,
                        checkAlertLevel,
                        primaryCheckIssueCode,
                        checkSummaryMessage,
                        checkRecommendedAction),
                metrics.dominantSeverity(),
                snapshot.failingChecks().isEmpty(),
                hasFailingChecksBySeverity(snapshot.failingChecks(), "critical"),
                hasFailingChecksBySeverity(snapshot.failingChecks(), "warning"),
                hasFailingChecksBySeverity(snapshot.failingChecks(), "info"),
                Map.of(
                        "critical", metrics.failingCriticalCheckCount(),
                        "warning", metrics.failingWarningCheckCount(),
                        "info", metrics.failingInfoCheckCount()),
                metrics.primaryFailingCheckName(),
                metrics.primaryFailingCheckCode(),
                metrics.primaryFailingCheckSeverity(),
                metrics.primaryFailingCheckMessage(),
                snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("name", "")))
                        .toList(),
                snapshot.failingChecks().stream()
                        .map(check -> String.valueOf(check.getOrDefault("code", "")))
                        .toList(),
                Map.of(
                        "critical", metrics.criticalCheckCount(),
                        "warning", metrics.warningCheckCount(),
                        "info", metrics.infoCheckCount()));
    }

    private static Map<String, Object> buildCheckSummaryBadge(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics,
            String checkStatus,
            String checkAlertLevel,
            String primaryCheckIssueCode,
            String checkSummaryMessage,
            String checkRecommendedAction) {
        LinkedHashMap<String, Object> badge = new LinkedHashMap<>();
        badge.put("status", checkStatus);
        badge.put("alertLevel", checkAlertLevel);
        badge.put("primaryIssueCode", primaryCheckIssueCode);
        badge.put("primaryCheckName", metrics.primaryFailingCheckName());
        badge.put("primaryCheckMessage", metrics.primaryFailingCheckMessage());
        badge.put("primaryCheckSeverity", metrics.primaryFailingCheckSeverity());
        badge.put("score", snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size());
        badge.put("variant", bundleHealthBadgeVariant(checkStatus));
        badge.put("label", bundleHealthBadgeLabel(checkStatus));
        badge.put("token", bundleHealthBadgeToken(checkStatus));
        badge.put("checkStatus", snapshot.failingChecks().isEmpty() ? "pass" : "fail");
        badge.put("summaryMessage", checkSummaryMessage);
        badge.put("recommendedAction", checkRecommendedAction);
        badge.put("tooltip", checkSummaryMessage);
        return badge;
    }

    private static DiffusionOpdBundleCheckSummaryBadge toTypedBundleCheckSummaryBadge(
            ManifestHealthSnapshot snapshot,
            CheckMetrics metrics,
            String checkStatus,
            String checkAlertLevel,
            String primaryCheckIssueCode,
            String checkSummaryMessage,
            String checkRecommendedAction) {
        return new DiffusionOpdBundleCheckSummaryBadge(
                checkStatus,
                checkAlertLevel,
                primaryCheckIssueCode,
                metrics.primaryFailingCheckName(),
                metrics.primaryFailingCheckMessage(),
                metrics.primaryFailingCheckSeverity(),
                snapshot.checks().isEmpty() ? 1.0d : (double) metrics.passingCheckCount() / (double) snapshot.checks().size(),
                bundleHealthBadgeVariant(checkStatus),
                bundleHealthBadgeLabel(checkStatus),
                bundleHealthBadgeToken(checkStatus),
                snapshot.failingChecks().isEmpty() ? "pass" : "fail",
                checkSummaryMessage,
                checkRecommendedAction,
                checkSummaryMessage);
    }

    private static DiffusionOpdBundleHealthCheck toTypedBundleHealthCheck(Map<String, Object> check) {
        return new DiffusionOpdBundleHealthCheck(
                String.valueOf(check.getOrDefault("name", "")),
                Boolean.TRUE.equals(check.get("passed")),
                String.valueOf(check.getOrDefault("severity", "")),
                String.valueOf(check.getOrDefault("code", "")),
                String.valueOf(check.getOrDefault("message", "")));
    }

    private static DiffusionOpdBundleGroupedCount toTypedGroupedCount(Map<String, Object> row) {
        Object countValue = row.get("count");
        int count = countValue instanceof Number number ? number.intValue() : 0;
        Object namesValue = row.get("names");
        List<String> names = namesValue instanceof List<?> list
                ? list.stream().map(String::valueOf).toList()
                : List.of();
        return new DiffusionOpdBundleGroupedCount(
                String.valueOf(row.getOrDefault("groupBy", "")),
                String.valueOf(row.getOrDefault("value", "")),
                count,
                names);
    }

    private static CheckMetrics collectCheckMetrics(ManifestHealthSnapshot snapshot) {
        int passingCheckCount = snapshot.checks().size() - snapshot.failingChecks().size();
        int criticalCheckCount = countChecksBySeverity(snapshot.checks(), "critical");
        int warningCheckCount = countChecksBySeverity(snapshot.checks(), "warning");
        int infoCheckCount = countChecksBySeverity(snapshot.checks(), "info");
        int failingCriticalCheckCount = countChecksBySeverity(snapshot.failingChecks(), "critical");
        int failingWarningCheckCount = countChecksBySeverity(snapshot.failingChecks(), "warning");
        int failingInfoCheckCount = countChecksBySeverity(snapshot.failingChecks(), "info");
        String primaryFailingCheckName = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("name", ""));
        String primaryFailingCheckCode = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("code", ""));
        String primaryFailingCheckSeverity = snapshot.failingChecks().isEmpty()
                ? "none"
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("severity", ""));
        String primaryFailingCheckMessage = snapshot.failingChecks().isEmpty()
                ? ""
                : String.valueOf(snapshot.failingChecks().get(0).getOrDefault("message", ""));
        return new CheckMetrics(
                passingCheckCount,
                criticalCheckCount,
                warningCheckCount,
                infoCheckCount,
                failingCriticalCheckCount,
                failingWarningCheckCount,
                failingInfoCheckCount,
                primaryFailingCheckName,
                primaryFailingCheckCode,
                primaryFailingCheckSeverity,
                primaryFailingCheckMessage,
                dominantCheckSeverity(criticalCheckCount, warningCheckCount, infoCheckCount));
    }

    private static String classifyBundleHealth(int totalFiles, int missingFileCount) {
        if (missingFileCount <= 0) {
            return "healthy";
        }
        if (missingFileCount >= totalFiles && totalFiles > 0) {
            return "broken";
        }
        return "degraded";
    }

    private static String buildBundleHealthSummary(String status, int totalFiles, int missingFileCount) {
        return switch (status) {
            case "healthy" -> "Bundle is healthy; all " + totalFiles + " manifest files are present.";
            case "broken" -> "Bundle is broken; all " + missingFileCount + " manifest files are missing.";
            default -> "Bundle is degraded; " + missingFileCount + " of " + totalFiles + " manifest files are missing.";
        };
    }

    private static String recommendedBundleHealthAction(String status) {
        return switch (status) {
            case "healthy" -> "none";
            case "broken" -> "rebuild-bundle";
            default -> "repair-missing-artifacts";
        };
    }

    private static List<String> bundleHealthIssueCodes(String status, int missingFileCount) {
        if ("healthy".equals(status)) {
            return List.of("bundle.ok");
        }
        if ("broken".equals(status)) {
            return List.of("bundle.artifacts.missing_all", "bundle.health.broken");
        }
        if (missingFileCount > 0) {
            return List.of("bundle.artifacts.missing_partial", "bundle.health.degraded");
        }
        return List.of("bundle.health.unknown");
    }

    private static String bundleHealthAlertLevel(String status) {
        return switch (status) {
            case "healthy" -> "info";
            case "broken" -> "critical";
            default -> "warning";
        };
    }

    private static String bundleHealthBadgeVariant(String status) {
        return switch (status) {
            case "healthy" -> "success";
            case "broken" -> "danger";
            default -> "warning";
        };
    }

    private static String bundleHealthBadgeLabel(String status) {
        return switch (status) {
            case "healthy" -> "Healthy";
            case "broken" -> "Broken";
            default -> "Degraded";
        };
    }

    private static String bundleHealthBadgeToken(String status) {
        return switch (status) {
            case "healthy" -> "bundle-health-healthy";
            case "broken" -> "bundle-health-broken";
            default -> "bundle-health-degraded";
        };
    }

    private static List<Map<String, Object>> buildBundleHealthChecks(
            boolean outputDirectoryConfigured,
            boolean outputDirectoryExists,
            int totalFiles,
            int missingFileCount,
            String status) {
        List<Map<String, Object>> checks = new ArrayList<>();
        checks.add(healthCheck(
                "outputDirectory",
                outputDirectoryConfigured && outputDirectoryExists,
                outputDirectoryConfigured ? (outputDirectoryExists ? "info" : "critical") : "warning",
                outputDirectoryConfigured
                        ? (outputDirectoryExists ? "bundle.output_directory.ok" : "bundle.output_directory.missing")
                        : "bundle.output_directory.unset",
                outputDirectoryConfigured
                        ? (outputDirectoryExists ? "Output directory exists." : "Output directory is missing.")
                        : "Output directory is not configured."));
        checks.add(healthCheck(
                "artifactsComplete",
                missingFileCount == 0,
                bundleHealthAlertLevel(status),
                missingFileCount == 0
                        ? "bundle.artifacts.complete"
                        : (missingFileCount >= totalFiles && totalFiles > 0
                                ? "bundle.artifacts.missing_all"
                                : "bundle.artifacts.missing_partial"),
                missingFileCount == 0
                        ? "All manifest artifacts are present."
                        : (missingFileCount >= totalFiles && totalFiles > 0
                                ? "All manifest artifacts are missing."
                                : "Some manifest artifacts are missing.")));
        return List.copyOf(checks);
    }

    private static Map<String, Object> healthCheck(
            String name,
            boolean passed,
            String severity,
            String code,
            String message) {
        LinkedHashMap<String, Object> check = new LinkedHashMap<>();
        check.put("name", name);
        check.put("passed", passed);
        check.put("severity", severity);
        check.put("code", code);
        check.put("message", message);
        return Map.copyOf(check);
    }

    private static int countChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return (int) checks.stream()
                .filter(check -> severity.equalsIgnoreCase(String.valueOf(check.getOrDefault("severity", ""))))
                .count();
    }

    private static boolean hasFailingChecksBySeverity(List<Map<String, Object>> checks, String severity) {
        return checks.stream()
                .anyMatch(check -> !Boolean.TRUE.equals(check.get("passed"))
                        && severity.equalsIgnoreCase(String.valueOf(check.getOrDefault("severity", ""))));
    }

    private static String dominantCheckSeverity(int criticalCheckCount, int warningCheckCount, int infoCheckCount) {
        if (criticalCheckCount >= warningCheckCount && criticalCheckCount >= infoCheckCount && criticalCheckCount > 0) {
            return "critical";
        }
        if (warningCheckCount >= infoCheckCount && warningCheckCount > 0) {
            return "warning";
        }
        if (infoCheckCount > 0) {
            return "info";
        }
        return "none";
    }

    private static Map<String, Object> summarizeManifestBundle(DiffusionOpdBundleManifest manifest) {
        return summarizeManifestBundle(manifest, "");
    }

    private static Map<String, Object> summarizeManifestBundle(DiffusionOpdBundleManifest manifest, String optionsExpression) {
        Map<String, String> options = parseManifestFilters(optionsExpression);
        LinkedHashMap<String, Object> summary = new LinkedHashMap<>(toTypedBundleSummary(manifest, options).toMap());
        if (options.containsKey("top")) {
            summary.put("top", options.get("top"));
        }
        return Map.copyOf(summary);
    }

    private static List<String> findMissingManifestFiles(DiffusionOpdBundleManifest manifest, List<Map<String, Object>> files) {
        return findMissingManifestFileEntries(manifest, files).stream()
                .map(entry -> String.valueOf(entry.getOrDefault("name", "")))
                .toList();
    }

    private static List<Map<String, Object>> findMissingManifestFileEntries(DiffusionOpdBundleManifest manifest, List<Map<String, Object>> files) {
        if (manifest.outputDirectory().isBlank()) {
            return List.of();
        }
        Path outputDirectoryPath = Path.of(manifest.outputDirectory());
        List<Map<String, Object>> missing = new ArrayList<>();
        for (Map<String, Object> file : files) {
            Object name = file.get("name");
            if (name == null || String.valueOf(name).isBlank()) {
                continue;
            }
            if (!Files.exists(outputDirectoryPath.resolve(String.valueOf(name)))) {
                missing.add(file);
            }
        }
        return List.copyOf(missing);
    }

    private static List<Map<String, Object>> summarizeMissingEntriesByField(
            List<Map<String, Object>> missingEntries,
            String field) {
        LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
        for (Map<String, Object> entry : missingEntries) {
            String value = String.valueOf(entry.getOrDefault(field, ""));
            counts.merge(value, 1, Integer::sum);
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            LinkedHashMap<String, Object> row = new LinkedHashMap<>();
            row.put("field", field);
            row.put("value", entry.getKey());
            row.put("count", entry.getValue());
            rows.add(Map.copyOf(row));
        }
        rows.sort(Comparator.comparing(row -> ((Number) row.getOrDefault("count", 0)).intValue(), Comparator.reverseOrder()));
        return List.copyOf(rows);
    }

    private static Object summarizeManifestFiles(DiffusionOpdBundleManifest manifest, String filterExpression) {
        ManifestFileQuery query = parseManifestFileQuery(filterExpression);
        Map<String, String> filters = query.filters();
        String groupBy = query.groupBy();
        List<Map<String, Object>> matches = filterManifestFilesList(manifest, filterExpression);
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
        sortManifestSummaryRows(rows, filters);
        rows = limitManifestSummaryRows(rows, filters);
        return List.copyOf(rows);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> castSummaryRows(Object value) {
        if (value instanceof List<?> list) {
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object element : list) {
                if (element instanceof Map<?, ?> entry) {
                    rows.add(normalizeMap(entry));
                }
            }
            return List.copyOf(rows);
        }
        return List.of();
    }

    private static List<Map<String, Object>> filterManifestFilesList(DiffusionOpdBundleManifest manifest, String filterExpression) {
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

    private static List<Map<String, Object>> manifestGeneratedFileEntries(DiffusionOpdBundleManifest manifest) {
        return manifest.generatedFiles().stream()
                .map(DiffusionOpdBundleGeneratedFile::toMap)
                .toList();
    }

    private static DiffusionOpdBundleGeneratedFile resolveTypedManifestFileEntry(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        Map<String, Object> entry = resolveManifestFileEntry(manifest, requested);
        return entry.isEmpty() ? null : DiffusionOpdBundleGeneratedFile.fromMap(entry);
    }

    private static Map<String, String> parseManifestFilters(String filterExpression) {
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

    private static ManifestFileQuery parseManifestFileQuery(String filterExpression) {
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

    private static void sortManifestSummaryRows(List<Map<String, Object>> rows, Map<String, String> filters) {
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

    private static List<Map<String, Object>> limitManifestSummaryRows(
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

    private static Object findManifestFile(DiffusionOpdBundleManifest manifest, String requested) {
        return findManifestFileEntryByName(manifestGeneratedFileEntries(manifest), requested);
    }

    private static Object loadManifestFileContent(DiffusionOpdBundleManifest manifest, String requested) {
        DiffusionOpdBundleGeneratedFile entry = resolveTypedManifestFileEntry(manifest, requested);
        if (entry == null) {
            return Map.of();
        }
        if (manifest.outputDirectory().isBlank() || entry.name().isBlank()) {
            return Map.of();
        }
        Path filePath = Path.of(manifest.outputDirectory()).resolve(entry.name());
        return loadBundleFileContent(filePath, entry.format());
    }

    private static Object loadTypedBundleFileContent(
            DiffusionOpdBundleManifest manifest,
            DiffusionOpdBundleGeneratedFile entry) {
        if (manifest.outputDirectory().isBlank() || entry.name().isBlank()) {
            return Map.of();
        }
        Path filePath = Path.of(manifest.outputDirectory()).resolve(entry.name());
        return loadBundleFileContent(filePath, entry.format());
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

    private static Object loadBundleFileContent(Path filePath, String format) {
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

    private static String renderValue(String name, Object value, String format, String columns) {
        return renderByFormat(new RenderRequest(name, value, format, columns));
    }

    private static String renderByFormat(RenderRequest request) {
        return switch (request.format()) {
            case "json" -> renderJson(request.value());
            case "table" -> renderTable(request.value(), request.columns());
            case "csv" -> renderCsv(request.value(), request.columns());
            case "text" -> request.name() + "=" + request.value() + System.lineSeparator();
            default -> throw new IllegalArgumentException(
                    "Unknown format '" + request.format() + "'. Use text, json, table, or csv.");
        };
    }

    private static String renderTable(Object value, String columns) {
        RenderGrid grid = prepareRenderGrid(value, columns);
        if (grid.rows().isEmpty()) {
            return "(no rows)" + System.lineSeparator();
        }
        TableLayout layout = prepareTableLayout(grid);
        StringBuilder output = new StringBuilder();
        output.append(renderTableRow(layout.headers(), layout)).append(System.lineSeparator());
        output.append(renderTableRow(layout.separators(), layout)).append(System.lineSeparator());
        for (Map<String, Object> row : grid.rows()) {
            output.append(renderTableRow(tableRowValues(row, layout.headers()), layout))
                    .append(System.lineSeparator());
        }
        return output.toString();
    }

    private static TableLayout prepareTableLayout(RenderGrid grid) {
        Map<String, Integer> widths = new LinkedHashMap<>();
        for (String header : grid.headers()) {
            widths.put(header, header.length());
        }
        for (Map<String, Object> row : grid.rows()) {
            for (String header : grid.headers()) {
                widths.put(header, Math.max(widths.get(header), stringifyCell(row.get(header)).length()));
            }
        }
        List<String> separators = new ArrayList<>();
        for (String header : grid.headers()) {
            separators.add("-".repeat(widths.get(header)));
        }
        return new TableLayout(grid.headers(), widths, List.copyOf(separators));
    }

    private static List<String> tableRowValues(Map<String, Object> row, List<String> headers) {
        List<String> values = new ArrayList<>();
        for (String header : headers) {
            values.add(stringifyCell(row.get(header)));
        }
        return List.copyOf(values);
    }

    private static String renderTableRow(List<String> values, TableLayout layout) {
        List<String> padded = new ArrayList<>();
        for (int index = 0; index < values.size(); index++) {
            String header = layout.headers().get(index);
            padded.add(padRight(values.get(index), layout.widths().get(header)));
        }
        return "| " + String.join(" | ", padded) + " |";
    }

    private static String padRight(String value, int width) {
        if (value.length() >= width) {
            return value;
        }
        return value + " ".repeat(width - value.length());
    }

    private static String renderCsv(Object value, String columns) {
        RenderGrid grid = prepareRenderGrid(value, columns);
        if (grid.rows().isEmpty()) {
            return "";
        }
        CsvRenderPlan plan = prepareCsvRenderPlan(grid);
        StringBuilder output = new StringBuilder();
        output.append(plan.headerLine()).append(System.lineSeparator());
        for (String rowLine : plan.rowLines()) {
            output.append(rowLine).append(System.lineSeparator());
        }
        return output.toString();
    }

    private static CsvRenderPlan prepareCsvRenderPlan(RenderGrid grid) {
        List<String> rowLines = new ArrayList<>();
        for (Map<String, Object> row : grid.rows()) {
            rowLines.add(renderCsvRow(row, grid.headers()));
        }
        return new CsvRenderPlan(renderCsvHeader(grid.headers()), List.copyOf(rowLines));
    }

    private static String renderCsvHeader(List<String> headers) {
        return String.join(",", headers.stream().map(DiffusionOpdReportInspectorSupport::escapeCsv).toList());
    }

    private static String renderCsvRow(Map<String, Object> row, List<String> headers) {
        List<String> values = new ArrayList<>();
        for (String header : headers) {
            values.add(escapeCsv(stringifyCell(row.get(header))));
        }
        return String.join(",", values);
    }

    private static RenderGrid prepareRenderGrid(Object value, String columns) {
        List<Map<String, Object>> rows = prepareProjectedRows(value, columns);
        return buildRenderGrid(rows);
    }

    private static List<Map<String, Object>> prepareProjectedRows(Object value, String columns) {
        return projectRows(toRows(value), columns);
    }

    private static RenderGrid buildRenderGrid(List<Map<String, Object>> rows) {
        return new RenderGrid(rows, headers(rows));
    }

    private static List<String> headers(List<Map<String, Object>> rows) {
        return List.copyOf(collectHeaders(rows));
    }

    private static LinkedHashSet<String> collectHeaders(List<Map<String, Object>> rows) {
        LinkedHashSet<String> headers = new LinkedHashSet<>();
        for (Map<String, Object> row : rows) {
            headers.addAll(row.keySet());
        }
        return headers;
    }

    private static List<Map<String, Object>> toRows(Object value) {
        if (value instanceof List<?> list) {
            return listToRows(list);
        }
        if (value instanceof Map<?, ?> map) {
            return mapToRows(map);
        }
        return List.of(scalarRow(value));
    }

    private static List<Map<String, Object>> listToRows(List<?> list) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object element : list) {
            rows.add(elementToRow(element));
        }
        return rows;
    }

    private static List<Map<String, Object>> mapToRows(Map<?, ?> map) {
        List<Map<String, Object>> expanded = expandNestedValueRows(map);
        if (!expanded.isEmpty()) {
            return expanded;
        }
        return List.of(normalizeMap(map));
    }

    private static List<Map<String, Object>> expandNestedValueRows(Map<?, ?> map) {
        Object nestedValues = map.get("values");
        if (!(nestedValues instanceof List<?> list)) {
            return List.of();
        }
        Map<String, Object> base = baseMapWithoutValues(map);
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object element : list) {
            if (element instanceof Map<?, ?> entry) {
                rows.add(mergeRowMaps(base, normalizeMap(entry)));
            }
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> elementToRow(Object element) {
        if (element instanceof Map<?, ?> map) {
            return normalizeMap(map);
        }
        return scalarRow(element);
    }

    private static Map<String, Object> mergeRowMaps(
            Map<String, Object> base,
            Map<String, Object> additions) {
        Map<String, Object> row = new LinkedHashMap<>(base);
        row.putAll(additions);
        return Map.copyOf(row);
    }

    private static Map<String, Object> baseMapWithoutValues(Map<?, ?> map) {
        Map<String, Object> row = new LinkedHashMap<>();
        for (Map.Entry<?, ?> outer : map.entrySet()) {
            if (!"values".equals(String.valueOf(outer.getKey()))) {
                row.put(String.valueOf(outer.getKey()), outer.getValue());
            }
        }
        return Map.copyOf(row);
    }

    private static Map<String, Object> scalarRow(Object value) {
        return Map.of("value", value);
    }

    private static List<Map<String, Object>> projectRows(List<Map<String, Object>> rows, String columns) {
        if (columns == null || columns.isBlank()) {
            return rows;
        }
        ProjectionSpec spec = buildProjectionSpec(columns, rows);
        if (spec.requestedColumns().isEmpty()) {
            return rows;
        }
        List<Map<String, Object>> projected = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            projected.add(projectRow(row, spec.requestedColumns()));
        }
        return List.copyOf(projected);
    }

    private static ProjectionSpec buildProjectionSpec(String columns, List<Map<String, Object>> rows) {
        String expanded = expandColumnsPreset(columns.trim(), rows);
        return new ProjectionSpec(expanded, parseRequestedColumns(expanded));
    }

    private static List<String> parseRequestedColumns(String expandedColumns) {
        return List.of(expandedColumns.split(",")).stream()
                .map(String::trim)
                .filter(column -> !column.isEmpty())
                .toList();
    }

    private static Map<String, Object> projectRow(Map<String, Object> row, List<String> requestedColumns) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (String column : requestedColumns) {
            copy.put(column, resolveProjectedValue(row, column));
        }
        return Map.copyOf(copy);
    }

    private static String expandColumnsPreset(String columns, List<Map<String, Object>> rows) {
        if (rows.isEmpty()) {
            return columns;
        }
        String identityColumn = rows.get(0).containsKey("pair")
                ? "pair"
                : rows.get(0).containsKey("value") ? "value" : "field";
        boolean pairRows = "pair".equals(identityColumn);
        return switch (columns) {
            case "minimal" -> identityColumn + ",loss";
            case "compact" -> identityColumn + ",count,loss";
            case "latest" -> identityColumn + ",count,loss,latestRound,latestLoss";
            case "leaderboard" -> identityColumn + ",count,loss,latestRound";
            case "details" -> identityColumn
                    + ",count,loss,latestRound,latestLoss,latestTeacher,latestTask,latestStage,top1Round,top1Loss,first1Round,first1Loss";
            case "compare" -> pairRows
                    ? "pair,count,loss,latestRound,latestLoss,latestTeacher,latestTask,latestStage,top1Round,top1Loss,first1Round,first1Loss"
                    : identityColumn + ",count,loss,latestRound,latestLoss,first1Round,first1Loss";
            default -> columns;
        };
    }

    private static String normalizeColumnsArgument(String columns) {
        if (columns == null) {
            return null;
        }
        String trimmed = columns.trim();
        return trimmed.isEmpty() || "_".equals(trimmed) ? null : trimmed;
    }

    private static CliOptions parseArgs(String[] args) {
        CliTokens tokens = tokenizeCliArgs(args);
        if (isBundleOutputOnlyInvocation(tokens)) {
            return new CliOptions(tokens.section(), "text", null, Path.of(tokens.third()));
        }
        if (isSupportedFormat(tokens.third())) {
            return parseExplicitFormatArgs(tokens);
        }
        if (tokens.third() != null && inferFormatFromPath(tokens.third()) != null) {
            return new CliOptions(tokens.section(), inferFormatFromPath(tokens.third()), null, Path.of(tokens.third()));
        }
        return parseDefaultArgs(tokens);
    }

    private static CliTokens tokenizeCliArgs(String[] args) {
        return new CliTokens(
                args.length > 1 ? args[1] : "all",
                args.length > 2 ? args[2] : null,
                args.length > 3 ? args[3] : null,
                args.length > 4 ? args[4] : null);
    }

    private static boolean isBundleOutputOnlyInvocation(CliTokens tokens) {
        return tokens.bundleSection()
                && tokens.third() != null
                && !isSupportedFormat(tokens.third())
                && tokens.fourth() == null;
    }

    private static CliOptions parseExplicitFormatArgs(CliTokens tokens) {
        String normalizedFormat = tokens.third().toLowerCase(Locale.ROOT);
        if (tokens.bundleSection() && tokens.fourth() != null && tokens.fifth() == null) {
            return new CliOptions(tokens.section(), normalizedFormat, null, Path.of(tokens.fourth()));
        }
        String columns = normalizeColumnsArgument(tokens.fourth());
        if (columns == null && tokens.fourth() != null && tokens.fifth() == null && looksLikeOutputPath(tokens.fourth())) {
            return new CliOptions(tokens.section(), normalizedFormat, null, Path.of(tokens.fourth()));
        }
        return new CliOptions(
                tokens.section(),
                normalizedFormat,
                columns,
                tokens.fifth() == null ? null : Path.of(tokens.fifth()));
    }

    private static CliOptions parseDefaultArgs(CliTokens tokens) {
        return new CliOptions(
                tokens.section(),
                "text",
                normalizeColumnsArgument(tokens.fourth()),
                tokens.fifth() == null ? null : Path.of(tokens.fifth()));
    }

    private static boolean isSupportedFormat(String value) {
        if (value == null) {
            return false;
        }
        String normalized = value.toLowerCase(Locale.ROOT);
        return "text".equals(normalized)
                || "json".equals(normalized)
                || "table".equals(normalized)
                || "csv".equals(normalized);
    }

    private static boolean looksLikeOutputPath(String value) {
        return value != null && inferFormatFromPath(value) != null;
    }

    private static String inferFormatFromPath(String value) {
        if (value == null) {
            return null;
        }
        String normalized = normalizePathToken(value);
        if (matchesAnySuffix(normalized, ".json")) {
            return "json";
        }
        if (matchesAnySuffix(normalized, ".csv")) {
            return "csv";
        }
        if (matchesAnySuffix(normalized, ".table", ".md")) {
            return "table";
        }
        if (matchesAnySuffix(normalized, ".txt", ".log")) {
            return "text";
        }
        return null;
    }

    private static String normalizePathToken(String value) {
        return value.toLowerCase(Locale.ROOT);
    }

    private static boolean matchesAnySuffix(String value, String... suffixes) {
        for (String suffix : suffixes) {
            if (value.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    private record CliOptions(String section, String format, String columns, Path outputPath) {
    }

    private record CliTokens(String section, String third, String fourth, String fifth) {
        private boolean bundleSection() {
            return section != null && section.startsWith("bundle=");
        }
    }

    private record ReportSectionView(String sectionName, String format, Object value) {
    }

    private record OutputEmission(String rendered, Path outputPath) {
    }

    private record RenderRequest(String name, Object value, String format, String columns) {
    }

    private record RenderGrid(List<Map<String, Object>> rows, List<String> headers) {
    }

    private record ProjectionSpec(String expandedColumns, List<String> requestedColumns) {
    }

    private record TableLayout(
            List<String> headers,
            Map<String, Integer> widths,
            List<String> separators) {
    }

    private record CsvRenderPlan(String headerLine, List<String> rowLines) {
    }

    private record ManifestFileQuery(
            Map<String, String> filters,
            String groupBy,
            String pick,
            String sort,
            ManifestSortSpec sortSpec,
            Integer index,
            Integer top) {
    }

    private record ManifestSortSpec(String field, boolean descending) {
    }

    private static Object resolveProjectedValue(Map<String, Object> row, String column) {
        String resolvedColumn = resolveColumnAlias(column);
        if (isDirectProjectedColumn(resolvedColumn)) {
            return row.get(resolvedColumn);
        }
        return descendProjectedPath(row, resolvedColumn);
    }

    private static String resolveColumnAlias(String column) {
        return COLUMN_ALIASES.getOrDefault(column, column);
    }

    private static boolean isDirectProjectedColumn(String resolvedColumn) {
        return !resolvedColumn.contains(".") && !resolvedColumn.contains("[");
    }

    private static Object descendProjectedPath(Map<String, Object> row, String resolvedColumn) {
        Object current = row;
        for (String part : resolvedColumn.split("\\.")) {
            current = descend(current, part);
            if (current == null) {
                return null;
            }
        }
        return current;
    }

    private static Object descend(Object current, String part) {
        int bracket = part.indexOf('[');
        String key = bracket >= 0 ? part.substring(0, bracket) : part;
        if (!(current instanceof Map<?, ?> map)) {
            return null;
        }
        Object next = map.get(key);
        if (next == null) {
            return null;
        }
        if (bracket < 0) {
            return next;
        }
        int close = part.indexOf(']', bracket);
        if (close < 0) {
            return null;
        }
        int index;
        try {
            index = Integer.parseInt(part.substring(bracket + 1, close));
        } catch (NumberFormatException exception) {
            return null;
        }
        if (!(next instanceof List<?> list) || index < 0 || index >= list.size()) {
            return null;
        }
        return list.get(index);
    }

    private static Map<String, Object> normalizeMap(Map<?, ?> map) {
        Map<String, Object> normalized = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            normalized.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(normalized);
    }

    private static String stringifyCell(Object value) {
        if (value == null) {
            return "";
        }
        if (isSimpleCellValue(value)) {
            return String.valueOf(value);
        }
        return stringifyStructuredCell(value);
    }

    private static boolean isSimpleCellValue(Object value) {
        return value instanceof Number || value instanceof Boolean || value instanceof CharSequence;
    }

    private static String stringifyStructuredCell(Object value) {
        try {
            return OBJECT_MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException exception) {
            return String.valueOf(value);
        }
    }

    private static String escapeCsv(String value) {
        boolean needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n");
        String escaped = value.replace("\"", "\"\"");
        return needsQuotes ? "\"" + escaped + "\"" : escaped;
    }

    private static String renderJson(Object value) {
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
            tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport report,
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
            tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport report,
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
        String rendered = renderValue(outputName, value, effectiveFormat, effectiveColumns);
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
        writeOutput(bundleDir.resolve("manifest.json"), renderJson(Map.of(
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
                normalizeColumnsArgument(token.columns()),
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
        if (!isSupportedFormat(normalized)) {
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

    private record ManifestHealthSnapshot(
            String outputDirectory,
            boolean outputDirectoryConfigured,
            boolean outputDirectoryExists,
            int totalFiles,
            int existingFileCount,
            int missingFileCount,
            boolean healthy,
            double healthScore,
            String healthStatus,
            String alertLevel,
            List<String> issueCodes,
            String primaryIssueCode,
            String summaryMessage,
            String recommendedAction,
            List<Map<String, Object>> checks,
            List<Map<String, Object>> failingChecks,
            List<String> missingFiles,
            List<Map<String, Object>> missingSections,
            List<Map<String, Object>> missingFormats) {
    }

    private record CheckMetrics(
            int passingCheckCount,
            int criticalCheckCount,
            int warningCheckCount,
            int infoCheckCount,
            int failingCriticalCheckCount,
            int failingWarningCheckCount,
            int failingInfoCheckCount,
            String primaryFailingCheckName,
            String primaryFailingCheckCode,
            String primaryFailingCheckSeverity,
            String primaryFailingCheckMessage,
            String dominantSeverity) {
    }

    private record ManifestSelection(String command, String argument, String fallbackSection) {
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
