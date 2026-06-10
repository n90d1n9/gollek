///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS com.fasterxml.jackson.core:jackson-databind:2.20.1

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * JBang compare inspector for recursive-reasoning report artifacts.
 */
public class trainer_recursive_reasoning_compare_inspector {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String REPORT_FILE_NAME = "recursive-reasoning-report.json";
    private static final String RESULT_FILE_NAME = "recursive-reasoning-compare-result.json";
    private static final String BUNDLE_MANIFEST_FILE_NAME = "manifest.json";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass either a compare-result artifact or a baseline/current report pair.");
        }
        CliOptions options = parseArgs(args);
        String format = normalizeFormat(options.format(), options.outputPath());
        if (options.resultMode()) {
            Path inputPath = Path.of(args[0]);
            if (isBundleDirectory(inputPath)) {
                BundleInput bundle = resolveBundle(inputPath);
                String section = options.section();
                Object value = selectBundle(bundle, section);

                System.out.println("====================================================");
                System.out.println(" Gollek Recursive Reasoning Compare Inspector");
                System.out.println("====================================================");
                System.out.println("bundlePath=" + inputPath.toAbsolutePath().normalize());
                System.out.println("bundleManifest=" + bundle.manifestFile().toAbsolutePath().normalize());
                System.out.println("section=" + (section == null ? "bundleSummary" : section));
                System.out.println("format=" + format);

                String rendered = renderValue(value, format);
                if (options.outputPath() != null) {
                    writeOutput(options.outputPath(), rendered);
                    System.out.println("wroteOutput=" + options.outputPath().toAbsolutePath().normalize());
                    return;
                }
                System.out.print(rendered);
                return;
            }

            Map<String, Object> result = resolveResult(inputPath);
            String section = options.section();
            Object value = selectResult(result, section);

            System.out.println("====================================================");
            System.out.println(" Gollek Recursive Reasoning Compare Inspector");
            System.out.println("====================================================");
            System.out.println("resultPath=" + Path.of(args[0]).toAbsolutePath().normalize());
            System.out.println("section=" + (section == null ? "resultOverview" : section));
            System.out.println("format=" + format);

            String rendered = renderValue(value, format);
            if (options.outputPath() != null) {
                writeOutput(options.outputPath(), rendered);
                System.out.println("wroteOutput=" + options.outputPath().toAbsolutePath().normalize());
                return;
            }
            System.out.print(rendered);
            return;
        }

        if (args.length < 2) {
            throw new IllegalArgumentException(
                    "Pass the baseline report path and current report path as the first two arguments.");
        }

        InputBundle baseline = resolveInput(Path.of(args[0]));
        InputBundle current = resolveInput(Path.of(args[1]));
        String section = options.section();
        Object value = select(baseline, current, section);

        if ("bundle=standard".equalsIgnoreCase(section) && options.outputPath() != null) {
            writeBundleArtifact(asMap(value), options.outputPath());
            System.out.println("====================================================");
            System.out.println(" Gollek Recursive Reasoning Compare Inspector");
            System.out.println("====================================================");
            System.out.println("baselinePath=" + baseline.inputPath().toAbsolutePath().normalize());
            System.out.println("currentPath=" + current.inputPath().toAbsolutePath().normalize());
            System.out.println("baselineReportFile=" + baseline.reportFile().toAbsolutePath().normalize());
            System.out.println("currentReportFile=" + current.reportFile().toAbsolutePath().normalize());
            System.out.println("section=" + section);
            System.out.println("format=" + format);
            System.out.println("wroteOutput=" + options.outputPath().toAbsolutePath().normalize());
            return;
        }

        System.out.println("====================================================");
        System.out.println(" Gollek Recursive Reasoning Compare Inspector");
        System.out.println("====================================================");
        System.out.println("baselinePath=" + baseline.inputPath().toAbsolutePath().normalize());
        System.out.println("currentPath=" + current.inputPath().toAbsolutePath().normalize());
        System.out.println("baselineReportFile=" + baseline.reportFile().toAbsolutePath().normalize());
        System.out.println("currentReportFile=" + current.reportFile().toAbsolutePath().normalize());
        System.out.println("section=" + (section == null ? "overview" : section));
        System.out.println("format=" + format);

        String rendered = renderValue(value, format);
        if (options.outputPath() != null) {
            writeOutput(options.outputPath(), rendered);
            System.out.println("wroteOutput=" + options.outputPath().toAbsolutePath().normalize());
            return;
        }
        System.out.print(rendered);
    }

    private static InputBundle resolveInput(Path inputPath) {
        Path reportFile = inputPath;
        if (Files.isDirectory(inputPath)) {
            reportFile = inputPath.resolve(REPORT_FILE_NAME);
        }
        if (!Files.isRegularFile(reportFile)) {
            throw new IllegalArgumentException(
                    "Recursive reasoning report not found: " + reportFile.toAbsolutePath().normalize());
        }
        return new InputBundle(inputPath, reportFile, loadJsonMap(reportFile));
    }

    private static Map<String, Object> resolveResult(Path inputPath) {
        Path resultFile = inputPath;
        if (Files.isDirectory(inputPath)) {
            resultFile = inputPath.resolve(RESULT_FILE_NAME);
        }
        if (!Files.isRegularFile(resultFile)) {
            throw new IllegalArgumentException(
                    "Recursive reasoning compare result not found: " + resultFile.toAbsolutePath().normalize());
        }
        return loadJsonMap(resultFile);
    }

    private static boolean isBundleDirectory(Path inputPath) {
        return Files.isDirectory(inputPath) && Files.isRegularFile(inputPath.resolve(BUNDLE_MANIFEST_FILE_NAME));
    }

    private static BundleInput resolveBundle(Path inputPath) {
        Path manifestFile = inputPath.resolve(BUNDLE_MANIFEST_FILE_NAME);
        if (!Files.isRegularFile(manifestFile)) {
            throw new IllegalArgumentException(
                    "Recursive reasoning compare bundle manifest not found: " + manifestFile.toAbsolutePath().normalize());
        }
        return new BundleInput(inputPath, manifestFile, loadJsonMap(manifestFile));
    }

    private static Object select(InputBundle baseline, InputBundle current, String section) {
        String normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        if ("result=standard".equals(normalized)) {
            return buildResultArtifact(baseline, current);
        }
        if ("bundle=standard".equals(normalized)) {
            return buildBundleArtifact(baseline, current);
        }
        if ("overview".equals(normalized)) {
            return overview(baseline, current, false);
        }
        if ("overview:ci".equals(normalized)) {
            return overview(baseline, current, true);
        }
        if ("comparison".equals(normalized)) {
            return comparison(baseline, current, false);
        }
        if ("comparison:ci".equals(normalized) || "ci".equals(normalized)) {
            return comparison(baseline, current, true);
        }
        if ("health".equals(normalized) || "status".equals(normalized)) {
            return health(baseline, current, false);
        }
        if ("health:ci".equals(normalized) || "status:ci".equals(normalized)) {
            return health(baseline, current, true);
        }
        if ("baseline".equals(normalized)) {
            return baseline.report();
        }
        if ("current".equals(normalized) || "report".equals(normalized)) {
            return current.report();
        }
        if (normalized.startsWith("delta:")) {
            return selectDelta(baseline.report(), current.report(), normalized.substring("delta:".length()));
        }
        throw new IllegalArgumentException(
                "Unknown section '" + section + "'. Use one of: overview, overview:ci, comparison, comparison:ci, health, health:ci, ci, baseline, current, report, result=standard, or delta:<field>.");
    }

    private static Object selectResult(Map<String, Object> result, String section) {
        String normalized = section == null ? "resultoverview" : section.toLowerCase(Locale.ROOT);
        if ("resultoverview".equals(normalized)) {
            return result.getOrDefault("overview", Map.of());
        }
        if ("resultoverview:ci".equals(normalized)) {
            return overviewResultCi(result);
        }
        if ("resultcomparison".equals(normalized)) {
            return result.getOrDefault("comparison", Map.of());
        }
        if ("resultcomparison:ci".equals(normalized)) {
            return comparisonResultCi(result);
        }
        if ("resulthealth".equals(normalized) || "resultstatus".equals(normalized)) {
            return result.getOrDefault("health", Map.of());
        }
        if ("resulthealth:ci".equals(normalized) || "resultstatus:ci".equals(normalized) || "resultci".equals(normalized)) {
            return healthResultCi(result);
        }
        if ("resultmanifest".equals(normalized)) {
            return result.getOrDefault("manifest", Map.of());
        }
        if ("resultsummary".equals(normalized)) {
            return resultSummary(result, false);
        }
        if ("resultsummary:ci".equals(normalized)) {
            return resultSummary(result, true);
        }
        if ("result".equals(normalized) || "report".equals(normalized)) {
            return result;
        }
        throw new IllegalArgumentException(
                "Unknown result section '" + section + "'. Use one of: resultOverview, resultOverview:ci, resultComparison, resultComparison:ci, resultHealth, resultHealth:ci, resultSummary, resultSummary:ci, resultManifest, result, or resultCi.");
    }

    private static Object selectBundle(BundleInput bundle, String section) {
        String normalized = section == null ? "bundlesummary" : section.toLowerCase(Locale.ROOT);
        if ("bundlesummary".equals(normalized)) {
            return bundleSummary(bundle);
        }
        if ("bundleoverview".equals(normalized)) {
            return loadBundleSection(bundle, "overview.json");
        }
        if ("bundleoverview:ci".equals(normalized)) {
            Map<String, Object> overview = asMap(loadBundleSection(bundle, "overview.json"));
            Map<String, Object> compact = new LinkedHashMap<>();
            compact.put("summary", "short");
            compact.put("comparison", asMap(overview.get("comparison")));
            compact.put("health", asMap(overview.get("health")));
            return compact;
        }
        if ("bundlemanifest".equals(normalized)) {
            return bundle.manifest();
        }
        if ("bundlefiles".equals(normalized) || "files".equals(normalized)) {
            return bundleFiles(bundle);
        }
        if ("bundlehealth".equals(normalized)) {
            return bundleHealth(bundle, false);
        }
        if ("bundlehealth:ci".equals(normalized) || "bundlestatus".equals(normalized)) {
            return bundleHealth(bundle, true);
        }
        if (normalized.startsWith("file:")) {
            return loadBundleSection(bundle, section.substring("file:".length()));
        }
        if (normalized.startsWith("loadfile:")) {
            String logical = normalized.substring("loadfile:".length());
            return loadBundleSection(bundle, switch (logical) {
                case "overview" -> "overview.json";
                case "comparison" -> "comparison.json";
                case "health" -> "health.json";
                case "summary" -> "summary.json";
                case "result" -> RESULT_FILE_NAME;
                default -> throw new IllegalArgumentException(
                        "Unknown bundle section '" + logical + "'. Use overview, comparison, health, summary, or result.");
            });
        }
        throw new IllegalArgumentException(
                "Unknown bundle section '" + section + "'. Use one of: bundleSummary, bundleOverview, bundleOverview:ci, bundleManifest, bundleFiles, bundleHealth, bundleHealth:ci, bundleStatus, file:<name>, or loadfile:<section>.");
    }

    private static Map<String, Object> overview(InputBundle baseline, InputBundle current, boolean ci) {
        Map<String, Object> comparison = comparison(baseline, current, ci);
        Map<String, Object> health = health(baseline, current, ci);
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("summary", ci ? "short" : "full");
        overview.put("comparison", comparison);
        overview.put("health", health);
        return overview;
    }

    private static Map<String, Object> comparison(InputBundle baseline, InputBundle current, boolean ci) {
        Map<String, Object> baselineReport = baseline.report();
        Map<String, Object> currentReport = current.report();
        Map<String, Object> baselineSummary = asMap(baselineReport.get("summary"));
        Map<String, Object> currentSummary = asMap(currentReport.get("summary"));
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("summary", ci ? "short" : "full");
        map.put("baselineTaskId", baselineReport.getOrDefault("taskId", ""));
        map.put("currentTaskId", currentReport.getOrDefault("taskId", ""));
        map.put("baselineSelectedStateId", selectedStateId(baselineReport, baselineSummary));
        map.put("currentSelectedStateId", selectedStateId(currentReport, currentSummary));
        map.put("baselineSelectedTrajectoryIndex", baselineReport.getOrDefault("selectedTrajectoryIndex", ""));
        map.put("currentSelectedTrajectoryIndex", currentReport.getOrDefault("selectedTrajectoryIndex", ""));
        map.put("baselineSelectedRewardScore", baselineReport.getOrDefault("selectedRewardScore", ""));
        map.put("currentSelectedRewardScore", currentReport.getOrDefault("selectedRewardScore", ""));
        map.put("rewardDelta", numericDelta(
                baselineReport.get("selectedRewardScore"),
                currentReport.get("selectedRewardScore")));
        map.put("cumulativeLogProbabilityDelta", numericDelta(
                baselineReport.get("selectedCumulativeLogProbability"),
                currentReport.get("selectedCumulativeLogProbability")));
        map.put("exploredTrajectoryCountDelta", numericDelta(
                baselineSummary.get("exploredTrajectoryCount"),
                currentSummary.get("exploredTrajectoryCount")));
        map.put("completedTrajectoryCountDelta", numericDelta(
                baselineSummary.get("completedTrajectoryCount"),
                currentSummary.get("completedTrajectoryCount")));
        return map;
    }

    private static Map<String, Object> health(InputBundle baseline, InputBundle current, boolean ci) {
        Map<String, Object> baselineReport = baseline.report();
        Map<String, Object> currentReport = current.report();
        Map<String, Object> baselineSummary = asMap(baselineReport.get("summary"));
        Map<String, Object> currentSummary = asMap(currentReport.get("summary"));
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("summary", ci ? "short" : "full");
        map.put("familyMatch", safeEquals(
                baselineReport.get("familyId"),
                currentReport.get("familyId")));
        map.put("reportVersionMatch", safeEquals(
                baselineReport.get("reportVersion"),
                currentReport.get("reportVersion")));
        map.put("baselineHasSelectedStateId", !String.valueOf(selectedStateId(baselineReport, baselineSummary)).isBlank());
        map.put("currentHasSelectedStateId", !String.valueOf(selectedStateId(currentReport, currentSummary)).isBlank());
        map.put("baselineReportFile", baseline.reportFile().toAbsolutePath().normalize().toString());
        map.put("currentReportFile", current.reportFile().toAbsolutePath().normalize().toString());
        return map;
    }

    private static Map<String, Object> buildResultArtifact(InputBundle baseline, InputBundle current) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("resultVersion", "1");
        result.put("manifest", resultManifest(baseline, current));
        result.put("overview", overview(baseline, current, false));
        result.put("comparison", comparison(baseline, current, false));
        result.put("health", health(baseline, current, false));
        result.put("summary", resultSummary(overview(baseline, current, false), comparison(baseline, current, false), health(baseline, current, false)));
        return result;
    }

    private static Map<String, Object> buildBundleArtifact(InputBundle baseline, InputBundle current) {
        Map<String, Object> result = buildResultArtifact(baseline, current);
        Map<String, Object> bundle = new LinkedHashMap<>();
        bundle.put("manifest", bundleManifest(baseline, current));
        bundle.put("result", result);
        bundle.put("overview", result.get("overview"));
        bundle.put("comparison", result.get("comparison"));
        bundle.put("health", result.get("health"));
        bundle.put("summary", result.get("summary"));
        return bundle;
    }

    private static Map<String, Object> bundleManifest(InputBundle baseline, InputBundle current) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("bundleVersion", "1");
        manifest.put("bundleType", "recursive-reasoning-compare");
        manifest.put("baselineReportFile", baseline.reportFile().toAbsolutePath().normalize().toString());
        manifest.put("currentReportFile", current.reportFile().toAbsolutePath().normalize().toString());
        manifest.put("files", Map.of(
                "manifest", BUNDLE_MANIFEST_FILE_NAME,
                "result", RESULT_FILE_NAME,
                "overview", "overview.json",
                "comparison", "comparison.json",
                "health", "health.json",
                "summary", "summary.json"));
        return manifest;
    }

    private static Map<String, Object> resultManifest(InputBundle baseline, InputBundle current) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("resultVersion", "1");
        manifest.put("resultType", "recursive-reasoning-compare");
        manifest.put("baselineReportFile", baseline.reportFile().toAbsolutePath().normalize().toString());
        manifest.put("currentReportFile", current.reportFile().toAbsolutePath().normalize().toString());
        manifest.put("baselineTaskId", baseline.report().getOrDefault("taskId", ""));
        manifest.put("currentTaskId", current.report().getOrDefault("taskId", ""));
        return manifest;
    }

    private static Map<String, Object> resultSummary(Map<String, Object> overview, Map<String, Object> comparison, Map<String, Object> health) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("rewardDelta", comparison.getOrDefault("rewardDelta", ""));
        summary.put("cumulativeLogProbabilityDelta", comparison.getOrDefault("cumulativeLogProbabilityDelta", ""));
        summary.put("familyMatch", health.getOrDefault("familyMatch", ""));
        summary.put("reportVersionMatch", health.getOrDefault("reportVersionMatch", ""));
        summary.put("baselineSelectedStateId", comparison.getOrDefault("baselineSelectedStateId", ""));
        summary.put("currentSelectedStateId", comparison.getOrDefault("currentSelectedStateId", ""));
        summary.put("overviewSection", "overview");
        return summary;
    }

    private static Map<String, Object> resultSummary(Map<String, Object> result, boolean ci) {
        Map<String, Object> summary = asMap(result.get("summary"));
        if (!ci) {
            return summary;
        }
        Map<String, Object> compact = new LinkedHashMap<>();
        compact.put("summary", "short");
        compact.put("rewardDelta", summary.getOrDefault("rewardDelta", ""));
        compact.put("cumulativeLogProbabilityDelta", summary.getOrDefault("cumulativeLogProbabilityDelta", ""));
        compact.put("familyMatch", summary.getOrDefault("familyMatch", ""));
        compact.put("reportVersionMatch", summary.getOrDefault("reportVersionMatch", ""));
        return compact;
    }

    private static Map<String, Object> overviewResultCi(Map<String, Object> result) {
        Map<String, Object> overview = asMap(result.get("overview"));
        Map<String, Object> compact = new LinkedHashMap<>();
        compact.put("summary", "short");
        compact.put("comparison", asMap(overview.get("comparison")));
        compact.put("health", asMap(overview.get("health")));
        return compact;
    }

    private static Map<String, Object> comparisonResultCi(Map<String, Object> result) {
        Map<String, Object> comparison = asMap(result.get("comparison"));
        Map<String, Object> compact = new LinkedHashMap<>();
        compact.put("summary", "short");
        compact.put("baselineSelectedStateId", comparison.getOrDefault("baselineSelectedStateId", ""));
        compact.put("currentSelectedStateId", comparison.getOrDefault("currentSelectedStateId", ""));
        compact.put("rewardDelta", comparison.getOrDefault("rewardDelta", ""));
        compact.put("cumulativeLogProbabilityDelta", comparison.getOrDefault("cumulativeLogProbabilityDelta", ""));
        return compact;
    }

    private static Map<String, Object> healthResultCi(Map<String, Object> result) {
        Map<String, Object> health = asMap(result.get("health"));
        Map<String, Object> compact = new LinkedHashMap<>();
        compact.put("summary", "short");
        compact.put("familyMatch", health.getOrDefault("familyMatch", ""));
        compact.put("reportVersionMatch", health.getOrDefault("reportVersionMatch", ""));
        compact.put("baselineHasSelectedStateId", health.getOrDefault("baselineHasSelectedStateId", ""));
        compact.put("currentHasSelectedStateId", health.getOrDefault("currentHasSelectedStateId", ""));
        return compact;
    }

    private static Map<String, Object> bundleSummary(BundleInput bundle) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("bundleVersion", bundle.manifest().getOrDefault("bundleVersion", "1"));
        summary.put("bundleType", bundle.manifest().getOrDefault("bundleType", ""));
        summary.put("fileCount", bundleFiles(bundle).size());
        summary.put("bundlePath", bundle.bundleDir().toAbsolutePath().normalize().toString());
        summary.put("files", bundleFiles(bundle));
        return summary;
    }

    private static Map<String, Object> bundleHealth(BundleInput bundle, boolean ci) {
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("summary", ci ? "short" : "full");
        health.put("hasManifest", Files.isRegularFile(bundle.manifestFile()));
        health.put("hasResult", Files.isRegularFile(bundle.bundleDir().resolve(RESULT_FILE_NAME)));
        health.put("hasOverview", Files.isRegularFile(bundle.bundleDir().resolve("overview.json")));
        health.put("hasComparison", Files.isRegularFile(bundle.bundleDir().resolve("comparison.json")));
        health.put("hasHealth", Files.isRegularFile(bundle.bundleDir().resolve("health.json")));
        health.put("hasSummary", Files.isRegularFile(bundle.bundleDir().resolve("summary.json")));
        return health;
    }

    private static Map<String, Object> loadBundleSection(BundleInput bundle, String fileName) {
        Path file = bundle.bundleDir().resolve(fileName);
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException("Bundle file not found: " + file.toAbsolutePath().normalize());
        }
        return loadJsonMap(file);
    }

    private static Map<String, String> bundleFiles(BundleInput bundle) {
        Map<String, String> files = new LinkedHashMap<>();
        files.put("manifest", bundle.bundleDir().resolve(BUNDLE_MANIFEST_FILE_NAME).toAbsolutePath().normalize().toString());
        files.put("result", bundle.bundleDir().resolve(RESULT_FILE_NAME).toAbsolutePath().normalize().toString());
        files.put("overview", bundle.bundleDir().resolve("overview.json").toAbsolutePath().normalize().toString());
        files.put("comparison", bundle.bundleDir().resolve("comparison.json").toAbsolutePath().normalize().toString());
        files.put("health", bundle.bundleDir().resolve("health.json").toAbsolutePath().normalize().toString());
        files.put("summary", bundle.bundleDir().resolve("summary.json").toAbsolutePath().normalize().toString());
        return files;
    }

    private static Object selectDelta(Map<String, Object> baseline, Map<String, Object> current, String key) {
        Map<String, Object> baselineSummary = asMap(baseline.get("summary"));
        Map<String, Object> currentSummary = asMap(current.get("summary"));
        return switch (key) {
            case "selectedrewardscore" -> numericDelta(
                    baseline.get("selectedRewardScore"),
                    current.get("selectedRewardScore"));
            case "selectedcumulativelogprobability" -> numericDelta(
                    baseline.get("selectedCumulativeLogProbability"),
                    current.get("selectedCumulativeLogProbability"));
            case "selectedtrajectoryindex" -> numericDelta(
                    baseline.get("selectedTrajectoryIndex"),
                    current.get("selectedTrajectoryIndex"));
            case "exploredtrajectorycount" -> numericDelta(
                    baselineSummary.get("exploredTrajectoryCount"),
                    currentSummary.get("exploredTrajectoryCount"));
            case "completedtrajectorycount" -> numericDelta(
                    baselineSummary.get("completedTrajectoryCount"),
                    currentSummary.get("completedTrajectoryCount"));
            default -> throw new IllegalArgumentException(
                    "Unknown delta field '" + key + "'. Use one of: selectedRewardScore, selectedCumulativeLogProbability, selectedTrajectoryIndex, exploredTrajectoryCount, completedTrajectoryCount.");
        };
    }

    private static Object selectedStateId(Map<String, Object> report, Map<String, Object> summary) {
        Object topLevel = report.get("selectedStateId");
        if (topLevel != null && !String.valueOf(topLevel).isBlank()) {
            return topLevel;
        }
        return summary.getOrDefault("selectedStateId", "");
    }

    private static Double numericDelta(Object baseline, Object current) {
        Double a = asDouble(baseline);
        Double b = asDouble(current);
        if (a == null || b == null) {
            return null;
        }
        return b - a;
    }

    private static Double asDouble(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static boolean safeEquals(Object left, Object right) {
        return String.valueOf(left).equals(String.valueOf(right));
    }

    private static String renderValue(Object value, String format) {
        try {
            if ("json".equals(format)) {
                return JSON.writeValueAsString(value) + System.lineSeparator();
            }
            if (value instanceof Map<?, ?> map) {
                StringBuilder sb = new StringBuilder();
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    sb.append(entry.getKey()).append('=').append(entry.getValue()).append(System.lineSeparator());
                }
                return sb.toString();
            }
            return String.valueOf(value) + System.lineSeparator();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to render value", e);
        }
    }

    private static String normalizeFormat(String format, Path outputPath) {
        if (format != null) {
            return format.toLowerCase(Locale.ROOT);
        }
        if (outputPath != null && outputPath.getFileName() != null
                && outputPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".json")) {
            return "json";
        }
        return "text";
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> loadJsonMap(Path path) {
        try {
            return JSON.readValue(path.toFile(), LinkedHashMap.class);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read JSON: " + path.toAbsolutePath().normalize(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        if (value instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return Map.of();
    }

    private static void writeOutput(Path outputPath, String rendered) {
        try {
            if (outputPath.getParent() != null) {
                Files.createDirectories(outputPath.getParent());
            }
            Files.writeString(outputPath, rendered);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to write output file: " + outputPath.toAbsolutePath().normalize(), e);
        }
    }

    private static void writeBundleArtifact(Map<String, Object> bundle, Path outputDir) {
        try {
            Files.createDirectories(outputDir);
            JSON.writeValue(outputDir.resolve(BUNDLE_MANIFEST_FILE_NAME).toFile(), bundle.get("manifest"));
            JSON.writeValue(outputDir.resolve(RESULT_FILE_NAME).toFile(), bundle.get("result"));
            JSON.writeValue(outputDir.resolve("overview.json").toFile(), bundle.get("overview"));
            JSON.writeValue(outputDir.resolve("comparison.json").toFile(), bundle.get("comparison"));
            JSON.writeValue(outputDir.resolve("health.json").toFile(), bundle.get("health"));
            JSON.writeValue(outputDir.resolve("summary.json").toFile(), bundle.get("summary"));
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to write bundle directory: " + outputDir.toAbsolutePath().normalize(), e);
        }
    }

    private static CliOptions parseArgs(String[] args) {
        if (args.length == 1 || (args.length >= 2 && isResultSelector(args[1]))) {
            String section = null;
            String format = null;
            Path outputPath = null;
            if (args.length >= 2) {
                if (isFormatToken(args[1])) {
                    format = args[1];
                } else {
                    section = args[1];
                }
            }
            if (args.length >= 3) {
                if (format == null && isFormatToken(args[2])) {
                    format = args[2];
                } else {
                    outputPath = Path.of(args[2]);
                }
            }
            if (args.length >= 4) {
                outputPath = Path.of(args[3]);
            }
            return new CliOptions(section, format, outputPath, true);
        }
        String section = null;
        String format = null;
        Path outputPath = null;
        if (args.length >= 3) {
            if (isFormatToken(args[2])) {
                format = args[2];
            } else {
                section = args[2];
            }
        }
        if (args.length >= 4) {
            if (format == null && isFormatToken(args[3])) {
                format = args[3];
            } else {
                outputPath = Path.of(args[3]);
            }
        }
        if (args.length >= 5) {
            outputPath = Path.of(args[4]);
        }
        return new CliOptions(section, format, outputPath, false);
    }

    private static boolean isFormatToken(String token) {
        String normalized = token.toLowerCase(Locale.ROOT);
        return "json".equals(normalized) || "text".equals(normalized);
    }

    private static boolean isResultSelector(String token) {
        String normalized = token.toLowerCase(Locale.ROOT);
        return normalized.startsWith("result") || normalized.startsWith("bundle") || normalized.startsWith("file:")
                || normalized.startsWith("loadfile:") || "files".equals(normalized);
    }

    private record InputBundle(Path inputPath, Path reportFile, Map<String, Object> report) {}

    private record BundleInput(Path bundleDir, Path manifestFile, Map<String, Object> manifest) {}

    private record CliOptions(String section, String format, Path outputPath, boolean resultMode) {}
}
