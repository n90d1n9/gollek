///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS com.fasterxml.jackson.core:jackson-databind:2.20.1

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * JBang inspector for byte-latent workflow manifest artifacts.
 */
public class trainer_byte_latent_workflow_inspector {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String BUNDLE_MANIFEST_FILE_NAME = "manifest.json";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the workflow manifest path as the first argument.");
        }
        CliOptions options = parseArgs(args);
        Path inputPath = Path.of(args[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        Path resolvedInput = resolveInput(inputPath);
        Map<String, Object> manifest = loadJsonMap(resolvedInput);
        if (isBundleSection(section)) {
            Path bundleDir = options.outputPath() != null
                    ? options.outputPath()
                    : resolvedInput.toAbsolutePath().normalize().getParent().resolve("byte-latent-workflow-bundle");
            Map<String, Object> bundleManifest = writeBundle(manifest, section, bundleDir);
            System.out.println("====================================================");
            System.out.println(" Gollek Byte-Latent Workflow Inspector");
            System.out.println("====================================================");
            System.out.println("inputPath=" + inputPath.toAbsolutePath().normalize());
            System.out.println("section=" + section);
            System.out.println("format=bundle");
            System.out.println("bundleDir=" + bundleDir.toAbsolutePath().normalize());
            System.out.print(renderJson(bundleManifest));
            return;
        }
        Object value = select(manifest, section);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Workflow Inspector");
        System.out.println("====================================================");
        System.out.println("inputPath=" + inputPath.toAbsolutePath().normalize());
        System.out.println("resolvedInput=" + resolvedInput.toAbsolutePath().normalize());
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

    private static Path resolveInput(Path inputPath) {
        if (Files.isDirectory(inputPath)) {
            Path nestedManifest = inputPath.resolve(BUNDLE_MANIFEST_FILE_NAME);
            if (Files.isRegularFile(nestedManifest)) {
                return nestedManifest;
            }
            throw new IllegalArgumentException("Bundle manifest not found: " + nestedManifest.toAbsolutePath().normalize());
        }
        if (Files.isRegularFile(inputPath)) {
            return inputPath;
        }
        throw new IllegalArgumentException("Inspector input not found: " + inputPath.toAbsolutePath().normalize());
    }

    private static boolean isBundleSection(String section) {
        return section != null && section.toLowerCase(Locale.ROOT).startsWith("bundle=");
    }

    private static Object select(Map<String, Object> manifest, String section) {
        String normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        if (isBundleManifest(manifest)) {
            return selectBundleManifest(manifest, normalized);
        }
        return switch (normalized) {
            case "overview" -> summarizeOverview(manifest);
            case "manifest", "report", "all" -> manifest;
            case "runs" -> manifest.getOrDefault("runs", List.of());
            case "commands", "inspectcommands" -> selectCommands(manifest, "");
            case "runcommands" -> selectRunCommands(manifest, "");
            case "runfields" -> selectRunFields(manifest, "");
            case "inspectfields" -> selectInspectFields(manifest, "");
            case "completedmodes" -> manifest.getOrDefault("completedModes", List.of());
            case "workflowbundledir" -> manifest.getOrDefault("workflowBundleDir", "");
            case "workflowbundlestatus" -> manifest.getOrDefault("workflowBundleStatus", "");
            case "bundlestatus", "status" -> summarizeMaterializedBundleOverview(manifest, "bundleoverview:ci");
            case "health" -> selectMaterializedBundle(manifest, "bundlehealth:ci");
            case "bundlemanifest" -> loadMaterializedBundleManifest(manifest);
            case "bundlefiles" -> selectMaterializedBundle(manifest, "files");
            case "bundlefilesummary" -> selectMaterializedBundle(manifest, "filessummary");
            case "bundlesummary" -> selectMaterializedBundle(manifest, "bundlesummary");
            case "bundlehealth" -> selectMaterializedBundle(manifest, "bundlehealth");
            case "historyrows" -> collectMetricRows(manifest, "historyRows");
            case "latesttrainloss" -> collectMetricRows(manifest, "latestTrainLoss");
            case "epochcount" -> collectMetricRows(manifest, "epochCount");
            default -> {
                if ("bundleoverview".equals(normalized) || normalized.startsWith("bundleoverview:")) {
                    yield summarizeMaterializedBundleOverview(manifest, normalized);
                }
                if (normalized.startsWith("bundlestatus:")) {
                    yield summarizeMaterializedBundleOverview(
                            manifest,
                            "bundleoverview:" + normalized.substring("bundlestatus:".length()));
                }
                if (normalized.startsWith("status:")) {
                    yield summarizeMaterializedBundleOverview(
                            manifest,
                            "bundleoverview:" + normalized.substring("status:".length()));
                }
                if (normalized.startsWith("health:")) {
                    yield selectMaterializedBundle(
                            manifest,
                            "bundlehealth:" + normalized.substring("health:".length()));
                }
                if (normalized.startsWith("bundlefiles:")) {
                    yield selectMaterializedBundle(manifest, "files:" + normalized.substring("bundlefiles:".length()));
                }
                if (normalized.startsWith("bundlefile:")) {
                    yield selectMaterializedBundle(manifest, "file:" + normalized.substring("bundlefile:".length()));
                }
                if (normalized.startsWith("bundleloadfile:")) {
                    yield selectMaterializedBundle(manifest, "loadfile:" + normalized.substring("bundleloadfile:".length()));
                }
                if (normalized.startsWith("bundlefilessummary:")) {
                    yield selectMaterializedBundle(manifest, "filessummary:" + normalized.substring("bundlefilessummary:".length()));
                }
                if (normalized.startsWith("run:")) {
                    yield findRun(manifest, normalized.substring("run:".length()));
                }
                if (normalized.startsWith("runcommands:")) {
                    yield selectRunCommands(manifest, normalized.substring("runcommands:".length()));
                }
                if (normalized.startsWith("runfields:")) {
                    yield selectRunFields(manifest, normalized.substring("runfields:".length()));
                }
                if (normalized.startsWith("inspectfields:")) {
                    yield selectInspectFields(manifest, normalized.substring("inspectfields:".length()));
                }
                if (normalized.startsWith("commands:") || normalized.startsWith("inspectcommands:")) {
                    String selector = normalized.startsWith("commands:")
                            ? normalized.substring("commands:".length())
                            : normalized.substring("inspectcommands:".length());
                    yield selectCommands(manifest, selector);
                }
                if (normalized.startsWith("delta:")) {
                    yield computeDelta(manifest, normalized.substring("delta:".length()));
                }
                throw new IllegalArgumentException(
                        "Unknown section '" + section + "'. Use one of: overview, manifest, runs, commands, runcommands, runcommands:<run>, runcommands:summary, runcommands:ci, runfields, runfields:<run>, runfields:summary, runfields:ci, inspectfields, inspectfields:summary, inspectfields:ci, completedModes, workflowBundleDir, workflowBundleStatus, status, bundleStatus, health, bundleOverview[:focus=health|files|all|ci], bundleManifest, bundleFiles, bundleSummary, bundleHealth, bundleLoadfile:section=runs, run:fresh, historyRows, latestTrainLoss, epochCount, delta:historyRows, delta:latestTrainLoss, delta:epochCount, or bundle=standard.");
            }
        };
    }

    private static boolean isBundleManifest(Map<String, Object> manifest) {
        return manifest.containsKey("bundleType") && manifest.containsKey("generatedFiles");
    }

    private static Object selectBundleManifest(Map<String, Object> manifest, String section) {
        if (section.startsWith("bundlehealth:")) {
            return summarizeBundleHealth(manifest, section.substring("bundlehealth:".length()));
        }
        if (section.startsWith("filessummary:")) {
            return summarizeBundleFiles(manifest, section.substring("filessummary:".length()));
        }
        if (section.startsWith("files:")) {
            return filterBundleFiles(manifest, section.substring("files:".length()));
        }
        if (section.startsWith("file:")) {
            return findBundleFile(manifest, section.substring("file:".length()));
        }
        if (section.startsWith("loadfile:")) {
            return loadBundleFile(manifest, section.substring("loadfile:".length()));
        }
        return switch (section) {
            case "overview", "manifest", "all", "report" -> manifest;
            case "files" -> manifest.getOrDefault("generatedFiles", List.of());
            case "filessummary" -> summarizeBundleFiles(manifest, "by=section");
            case "bundlesummary" -> summarizeBundle(manifest);
            case "bundlehealth", "bundlestatus", "status", "health" -> summarizeBundleHealth(manifest, "ci");
            default -> throw new IllegalArgumentException(
                    "Unknown bundle section '" + section + "'. Use one of: overview, manifest, files, files:section=<section>, files:name=<name>, filesSummary:by=section, filesSummary:by=format, bundleSummary, bundleHealth[:summary=short|ci], status, bundleStatus, health, file:<name>, or loadfile:section=<section>.");
        };
    }

    private static Map<String, Object> loadMaterializedBundleManifest(Map<String, Object> manifest) {
        return loadJsonMap(resolveMaterializedBundleManifestPath(manifest));
    }

    private static Map<String, Object> summarizeMaterializedBundleOverview(Map<String, Object> manifest, String selector) {
        BundleOverviewOptions options = parseBundleOverviewOptions(selector);
        Map<String, Object> overview = new LinkedHashMap<>();
        String bundleDir = String.valueOf(manifest.getOrDefault("workflowBundleDir", ""));
        String bundleStatus = String.valueOf(manifest.getOrDefault("workflowBundleStatus", ""));
        overview.put("workflowBundleDir", bundleDir);
        overview.put("workflowBundleStatus", bundleStatus);
        overview.put("bundleAvailable", "materialized".equalsIgnoreCase(bundleStatus));
        overview.put("focus", options.focus());
        overview.put("summary", options.summary());
        if (!"materialized".equalsIgnoreCase(bundleStatus)) {
            overview.put("summaryMessage", "Workflow bundle is not materialized for this manifest.");
            return overview;
        }
        Map<String, Object> bundleManifest = loadMaterializedBundleManifest(manifest);
        Map<String, Object> bundleSummary = summarizeBundle(bundleManifest);
        Map<String, Object> bundleHealth = summarizeBundleHealth(bundleManifest);
        if ("all".equals(options.focus()) || "files".equals(options.focus())) {
            overview.put("bundleType", bundleSummary.getOrDefault("bundleType", ""));
            overview.put("fileCount", bundleSummary.getOrDefault("fileCount", 0));
        }
        if ("all".equals(options.focus()) || "health".equals(options.focus())) {
            overview.put("missingFileCount", bundleHealth.getOrDefault("missingFileCount", 0));
            overview.put("healthStatus", bundleHealth.getOrDefault("status", ""));
            overview.put("summaryMessage", bundleHealth.getOrDefault("summaryMessage", ""));
        }
        return overview;
    }

    private static BundleOverviewOptions parseBundleOverviewOptions(String selector) {
        String normalized = selector == null ? "bundleoverview" : selector.toLowerCase(Locale.ROOT);
        if ("bundleoverview".equals(normalized)) {
            return new BundleOverviewOptions("all", "full");
        }
        String[] parts = normalized.split(":");
        String focus = "all";
        String summary = "full";
        for (int index = 1; index < parts.length; index++) {
            String part = parts[index].trim();
            if (part.isBlank()) {
                continue;
            }
            if ("ci".equals(part)) {
                focus = "health";
                summary = "short";
                continue;
            }
            if (part.startsWith("focus=")) {
                String value = part.substring("focus=".length()).trim();
                if ("all".equals(value) || "health".equals(value) || "files".equals(value)) {
                    focus = value;
                    continue;
                }
                throw new IllegalArgumentException(
                        "Unsupported bundleOverview focus '" + value + "'. Use focus=health, focus=files, or focus=all.");
            }
            if (part.startsWith("summary=")) {
                String value = part.substring("summary=".length()).trim();
                if ("full".equals(value) || "short".equals(value)) {
                    summary = value;
                    continue;
                }
                throw new IllegalArgumentException(
                        "Unsupported bundleOverview summary '" + value + "'. Use summary=full or summary=short.");
            }
            throw new IllegalArgumentException(
                    "Unsupported bundleOverview selector part '" + parts[index] + "'. Use bundleOverview:focus=health|files|all:summary=full|short or bundleOverview:ci.");
        }
        return new BundleOverviewOptions(focus, summary);
    }

    private static Object selectMaterializedBundle(Map<String, Object> manifest, String section) {
        return selectBundleManifest(loadMaterializedBundleManifest(manifest), section.toLowerCase(Locale.ROOT));
    }

    private static Path resolveMaterializedBundleManifestPath(Map<String, Object> manifest) {
        String bundleStatus = String.valueOf(manifest.getOrDefault("workflowBundleStatus", ""));
        String bundleDir = String.valueOf(manifest.getOrDefault("workflowBundleDir", ""));
        if (!"materialized".equalsIgnoreCase(bundleStatus)) {
            throw new IllegalArgumentException(
                    "Workflow bundle is not materialized for this manifest. Current workflowBundleStatus=" + bundleStatus);
        }
        if (bundleDir.isBlank()) {
            throw new IllegalArgumentException("Workflow manifest does not include workflowBundleDir.");
        }
        Path manifestPath = Path.of(bundleDir).resolve(BUNDLE_MANIFEST_FILE_NAME);
        if (!Files.isRegularFile(manifestPath)) {
            throw new IllegalArgumentException("Materialized workflow bundle manifest not found: "
                    + manifestPath.toAbsolutePath().normalize());
        }
        return manifestPath;
    }

    private static Map<String, Object> writeBundle(Map<String, Object> manifest, String section, Path bundleDir) {
        String normalized = section.toLowerCase(Locale.ROOT);
        if (!"bundle=standard".equals(normalized)) {
            throw new IllegalArgumentException("Unsupported bundle section '" + section + "'. Use bundle=standard.");
        }
        try {
            Files.createDirectories(bundleDir);
            List<Map<String, Object>> generatedFiles = new ArrayList<>();
            writeBundleEntry(bundleDir, generatedFiles, "overview.json", "overview", renderJson(summarizeOverview(manifest)));
            writeBundleEntry(bundleDir, generatedFiles, "runs.json", "runs", renderJson(manifest.getOrDefault("runs", List.of())));
            writeBundleEntry(bundleDir, generatedFiles, "delta-historyRows.json", "delta:historyRows", renderJson(computeDelta(manifest, "historyRows")));
            writeBundleEntry(bundleDir, generatedFiles, "delta-latestTrainLoss.json", "delta:latestTrainLoss", renderJson(computeDelta(manifest, "latestTrainLoss")));
            Map<String, Object> bundleManifest = new LinkedHashMap<>();
            bundleManifest.put("bundleType", "standard");
            bundleManifest.put("outputDirectory", bundleDir.toAbsolutePath().normalize().toString());
            bundleManifest.put("generatedFiles", generatedFiles);
            bundleManifest.put("fileCount", generatedFiles.size());
            bundleManifest.put("createdAt", manifest.getOrDefault("createdAt", ""));
            String rendered = renderJson(bundleManifest);
            Files.writeString(bundleDir.resolve("manifest.json"), rendered);
            return bundleManifest;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write bundle directory " + bundleDir + ".", exception);
        }
    }

    private static Map<String, Object> summarizeBundle(Map<String, Object> manifest) {
        List<Map<String, Object>> files = generatedFiles(manifest);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("bundleType", manifest.getOrDefault("bundleType", ""));
        summary.put("outputDirectory", manifest.getOrDefault("outputDirectory", ""));
        summary.put("fileCount", files.size());
        summary.put("sections", files.stream().map(file -> String.valueOf(file.getOrDefault("section", ""))).toList());
        summary.put("formats", files.stream().map(file -> String.valueOf(file.getOrDefault("format", ""))).distinct().toList());
        summary.put("createdAt", manifest.getOrDefault("createdAt", ""));
        return summary;
    }

    private static Map<String, Object> summarizeBundleHealth(Map<String, Object> manifest, String selector) {
        String summary = parseBundleHealthSummary(selector);
        List<Map<String, Object>> files = generatedFiles(manifest);
        String outputDirectory = String.valueOf(manifest.getOrDefault("outputDirectory", ""));
        List<String> missing = new ArrayList<>();
        for (Map<String, Object> file : files) {
            String name = String.valueOf(file.getOrDefault("name", ""));
            if (!outputDirectory.isBlank() && !Files.isRegularFile(Path.of(outputDirectory).resolve(name))) {
                missing.add(name);
            }
        }
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("bundleType", manifest.getOrDefault("bundleType", ""));
        health.put("outputDirectory", outputDirectory);
        health.put("expectedFileCount", files.size());
        health.put("missingFileCount", missing.size());
        health.put("missingFiles", List.copyOf(missing));
        health.put("status", missing.isEmpty() ? "healthy" : "degraded");
        health.put("summary", summary);
        health.put("summaryMessage", missing.isEmpty()
                ? "Workflow bundle is healthy; all manifest files are present."
                : "Workflow bundle is degraded; some manifest files are missing.");
        return health;
    }

    private static String parseBundleHealthSummary(String selector) {
        if (selector == null || selector.isBlank()) {
            return "full";
        }
        String[] parts = selector.toLowerCase(Locale.ROOT).split(":");
        String summary = "full";
        for (String rawPart : parts) {
            String part = rawPart.trim();
            if (part.isBlank()) {
                continue;
            }
            if ("ci".equals(part)) {
                summary = "short";
                continue;
            }
            if (part.startsWith("summary=")) {
                String value = part.substring("summary=".length()).trim();
                if ("full".equals(value) || "short".equals(value)) {
                    summary = value;
                    continue;
                }
                throw new IllegalArgumentException(
                        "Unsupported bundleHealth summary '" + value + "'. Use summary=full, summary=short, or ci.");
            }
            throw new IllegalArgumentException(
                    "Unsupported bundleHealth selector part '" + rawPart + "'. Use bundleHealth:summary=full|short or bundleHealth:ci.");
        }
        return summary;
    }

    private static List<Map<String, Object>> generatedFiles(Map<String, Object> manifest) {
        Object value = manifest.get("generatedFiles");
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof Map<?, ?> row) {
                Map<String, Object> copy = new LinkedHashMap<>();
                row.forEach((key, element) -> copy.put(String.valueOf(key), element));
                rows.add(copy);
            }
        }
        return List.copyOf(rows);
    }

    private static List<Map<String, Object>> filterBundleFiles(Map<String, Object> manifest, String selector) {
        String[] parts = selector.trim().split(":");
        String expectedSection = null;
        String expectedName = null;
        String sort = null;
        for (String rawPart : parts) {
            String part = rawPart.trim();
            if (part.isBlank()) {
                continue;
            }
            String normalized = part.toLowerCase(Locale.ROOT);
            if (normalized.startsWith("section=")) {
                expectedSection = normalized.substring("section=".length()).trim();
                continue;
            }
            if (normalized.startsWith("name=")) {
                expectedName = normalized.substring("name=".length()).trim();
                continue;
            }
            if (normalized.startsWith("sort=")) {
                sort = normalized.substring("sort=".length()).trim();
                continue;
            }
            throw new IllegalArgumentException("Unsupported files selector part '" + rawPart + "'. Use section=..., name=..., and optional sort=....");
        }
        List<Map<String, Object>> rows = new ArrayList<>(generatedFiles(manifest));
        if (expectedSection != null) {
            rows.removeIf(file -> !expectedSection.equals(String.valueOf(file.getOrDefault("section", "")).toLowerCase(Locale.ROOT)));
        }
        if (expectedName != null) {
            rows.removeIf(file -> !expectedName.equals(String.valueOf(file.getOrDefault("name", "")).toLowerCase(Locale.ROOT)));
        }
        rows.sort((left, right) -> compareFileRows(left, right, sort));
        return List.copyOf(rows);
    }

    private static List<Map<String, Object>> summarizeBundleFiles(Map<String, Object> manifest, String selector) {
        String[] parts = selector.trim().split(":");
        String groupBy = null;
        String sort = null;
        Integer top = null;
        for (String rawPart : parts) {
            String part = rawPart.trim().toLowerCase(Locale.ROOT);
            if (part.isBlank()) {
                continue;
            }
            if (part.startsWith("by=")) {
                String value = part.substring("by=".length());
                if ("section".equals(value) || "format".equals(value)) {
                    groupBy = value;
                    continue;
                }
                throw new IllegalArgumentException("Unsupported filesSummary group '" + value + "'. Use by=section or by=format.");
            }
            if (part.startsWith("sort=")) {
                sort = part.substring("sort=".length());
                continue;
            }
            if (part.startsWith("top=")) {
                top = Integer.parseInt(part.substring("top=".length()));
                continue;
            }
            throw new IllegalArgumentException("Unsupported filesSummary selector part '" + rawPart + "'. Use by=section|format, optional sort=..., and optional top=<n>.");
        }
        if (groupBy == null) {
            throw new IllegalArgumentException("Unsupported filesSummary selector '" + selector + "'. Use filesSummary:by=section or filesSummary:by=format.");
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map<String, Object> file : generatedFiles(manifest)) {
            String value = String.valueOf(file.getOrDefault(groupBy, ""));
            counts.put(value, counts.getOrDefault(value, 0) + 1);
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        counts.forEach((value, count) -> {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("value", value);
            row.put("count", count);
            rows.add(row);
        });
        rows.sort((left, right) -> compareSummaryRows(left, right, sort));
        if (top != null && top >= 0 && top < rows.size()) {
            rows = new ArrayList<>(rows.subList(0, top));
        }
        return List.copyOf(rows);
    }

    private static int compareSummaryRows(Map<String, Object> left, Map<String, Object> right, String sort) {
        String effectiveSort = (sort == null || sort.isBlank()) ? "value" : sort;
        boolean descending = effectiveSort.startsWith("-");
        String field = descending ? effectiveSort.substring(1) : effectiveSort;
        int comparison;
        if ("count".equals(field)) {
            comparison = Integer.compare(
                    Integer.parseInt(String.valueOf(left.getOrDefault("count", 0))),
                    Integer.parseInt(String.valueOf(right.getOrDefault("count", 0))));
        } else {
            comparison = String.valueOf(left.getOrDefault("value", ""))
                    .compareToIgnoreCase(String.valueOf(right.getOrDefault("value", "")));
        }
        return descending ? -comparison : comparison;
    }

    private static int compareFileRows(Map<String, Object> left, Map<String, Object> right, String sort) {
        String effectiveSort = (sort == null || sort.isBlank()) ? "name" : sort;
        boolean descending = effectiveSort.startsWith("-");
        String field = descending ? effectiveSort.substring(1) : effectiveSort;
        int comparison = switch (field) {
            case "section" -> String.valueOf(left.getOrDefault("section", ""))
                    .compareToIgnoreCase(String.valueOf(right.getOrDefault("section", "")));
            case "format" -> String.valueOf(left.getOrDefault("format", ""))
                    .compareToIgnoreCase(String.valueOf(right.getOrDefault("format", "")));
            case "name" -> String.valueOf(left.getOrDefault("name", ""))
                    .compareToIgnoreCase(String.valueOf(right.getOrDefault("name", "")));
            default -> throw new IllegalArgumentException("Unsupported files sort field '" + field + "'. Use sort=name, sort=section, or sort=format.");
        };
        return descending ? -comparison : comparison;
    }

    private static Map<String, Object> findBundleFile(Map<String, Object> manifest, String requested) {
        String normalized = requested.trim().toLowerCase(Locale.ROOT);
        for (Map<String, Object> file : generatedFiles(manifest)) {
            if (normalized.equals(String.valueOf(file.getOrDefault("name", "")).toLowerCase(Locale.ROOT))) {
                return file;
            }
        }
        throw new IllegalArgumentException("Bundle file not found: " + requested);
    }

    private static Object loadBundleFile(Map<String, Object> manifest, String selector) {
        Map<String, Object> file = resolveBundleFile(manifest, selector);
        String outputDirectory = String.valueOf(manifest.getOrDefault("outputDirectory", ""));
        if (outputDirectory.isBlank()) {
            throw new IllegalArgumentException("Bundle manifest does not include an outputDirectory.");
        }
        Path path = Path.of(outputDirectory).resolve(String.valueOf(file.getOrDefault("name", "")));
        try {
            return loadJsonMap(path);
        } catch (IllegalStateException exception) {
            throw new IllegalArgumentException("Failed to load bundle file: " + path.toAbsolutePath().normalize(), exception);
        }
    }

    private static Map<String, Object> resolveBundleFile(Map<String, Object> manifest, String selector) {
        String[] parts = selector.trim().split(":");
        String expectedSection = null;
        String pick = "first";
        Integer index = null;
        for (String rawPart : parts) {
            String part = rawPart.trim();
            if (part.isBlank()) {
                continue;
            }
            String normalized = part.toLowerCase(Locale.ROOT);
            if (normalized.startsWith("section=")) {
                expectedSection = normalized.substring("section=".length()).trim();
                continue;
            }
            if (normalized.startsWith("pick=")) {
                pick = normalized.substring("pick=".length()).trim();
                continue;
            }
            if (normalized.startsWith("index=")) {
                index = Integer.parseInt(normalized.substring("index=".length()).trim());
                continue;
            }
            return findBundleFile(manifest, part);
        }
        if (expectedSection != null) {
            List<Map<String, Object>> matches = generatedFiles(manifest).stream()
                    .filter(file -> expectedSection.equals(String.valueOf(file.getOrDefault("section", "")).toLowerCase(Locale.ROOT)))
                    .toList();
            if (matches.isEmpty()) {
                throw new IllegalArgumentException("Bundle section not found: " + expectedSection);
            }
            if (index != null) {
                if (index < 0 || index >= matches.size()) {
                    throw new IllegalArgumentException("Bundle section index out of range: " + index);
                }
                return matches.get(index);
            }
            if ("last".equals(pick)) {
                return matches.get(matches.size() - 1);
            }
            if ("first".equals(pick)) {
                return matches.get(0);
            }
            throw new IllegalArgumentException("Unsupported loadfile pick '" + pick + "'. Use pick=first, pick=last, or index=<n>.");
        }
        return findBundleFile(manifest, selector.trim());
    }

    private static void writeBundleEntry(
            Path bundleDir,
            List<Map<String, Object>> generatedFiles,
            String fileName,
            String section,
            String content) throws IOException {
        Files.writeString(bundleDir.resolve(fileName), content);
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("name", fileName);
        entry.put("section", section);
        entry.put("format", "json");
        generatedFiles.add(entry);
    }

    private static Map<String, Object> summarizeOverview(Map<String, Object> manifest) {
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("workflow", manifest.getOrDefault("workflow", ""));
        overview.put("mode", manifest.getOrDefault("mode", ""));
        overview.put("checkpointRoot", manifest.getOrDefault("checkpointRoot", ""));
        overview.put("workflowBundleDir", manifest.getOrDefault("workflowBundleDir", ""));
        overview.put("workflowBundleStatus", manifest.getOrDefault("workflowBundleStatus", ""));
        overview.put("createdAt", manifest.getOrDefault("createdAt", ""));
        overview.put("completedModes", manifest.getOrDefault("completedModes", List.of()));
        overview.put("runCount", runs(manifest).size());
        overview.put("historyRows", collectMetricRows(manifest, "historyRows"));
        overview.put("latestTrainLoss", collectMetricRows(manifest, "latestTrainLoss"));
        overview.put("deltaHistoryRows", computeDelta(manifest, "historyRows"));
        overview.put("deltaLatestTrainLoss", computeDelta(manifest, "latestTrainLoss"));
        return overview;
    }

    private static Object selectCommands(Map<String, Object> manifest, String selector) {
        String expectedScope = null;
        String expectedMode = null;
        String expectedName = null;
        String summary = null;
        if (selector != null && !selector.isBlank()) {
            for (String rawPart : selector.split(":")) {
                String part = rawPart.trim().toLowerCase(Locale.ROOT);
                if (part.isBlank()) {
                    continue;
                }
                if ("summary".equals(part)) {
                    summary = "full";
                    continue;
                }
                if ("ci".equals(part)) {
                    summary = "short";
                    continue;
                }
                if (part.startsWith("scope=")) {
                    expectedScope = part.substring("scope=".length()).trim();
                    continue;
                }
                if (part.startsWith("mode=")) {
                    expectedMode = part.substring("mode=".length()).trim();
                    continue;
                }
                if (part.startsWith("name=")) {
                    expectedName = part.substring("name=".length()).trim();
                    continue;
                }
            }
        }
        if (summary != null) {
            return summarizeCommands(manifest, expectedScope, expectedMode, expectedName, summary);
        }
        return collectCommandRows(manifest, selector);
    }

    private static Object selectRunCommands(Map<String, Object> manifest, String selector) {
        String normalized = selector == null ? "" : selector.trim().toLowerCase(Locale.ROOT);
        if (normalized.isBlank()) {
            return listRunCommandAliases(manifest);
        }
        if ("all".equals(normalized)) {
            return summarizeRunCommands(manifest, "full");
        }
        if ("summary".equals(normalized) || "ci".equals(normalized)) {
            return summarizeRunCommands(manifest, "ci".equals(normalized) ? "short" : "full");
        }
        return selectCommands(manifest, "ci:mode=" + normalized);
    }

    private static List<Map<String, Object>> listRunCommandAliases(Map<String, Object> manifest) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("mode", mode);
            row.put("alias", "runcommands:" + mode);
            row.put("ciAlias", "commands:ci:mode=" + mode);
            row.put("summaryAlias", "commands:summary:mode=" + mode);
            Object inspectRunCommands = run.get("inspectRunCommands");
            if (inspectRunCommands instanceof String text && !text.isBlank()) {
                row.put("command", text);
            }
            rows.add(row);
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> summarizeRunCommands(Map<String, Object> manifest, String summaryMode) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summaryType", "runcommands");
        summary.put("summary", summaryMode);
        List<String> modes = new ArrayList<>();
        List<String> aliases = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            modes.add(mode);
            aliases.add("runcommands:" + mode);
            Object inspectRunCommands = run.get("inspectRunCommands");
            if (inspectRunCommands instanceof String text && !text.isBlank()) {
                summary.put(mode, text);
            }
        }
        summary.put("modes", List.copyOf(modes));
        summary.put("aliases", List.copyOf(aliases));
        return summary;
    }

    private static Object selectRunFields(Map<String, Object> manifest, String selector) {
        String normalized = selector == null ? "" : selector.trim().toLowerCase(Locale.ROOT);
        if (normalized.isBlank()) {
            return listRunFieldAliases(manifest);
        }
        if ("all".equals(normalized) || "summary".equals(normalized)) {
            return summarizeRunFields(manifest, "full");
        }
        if ("ci".equals(normalized)) {
            return summarizeRunFields(manifest, "short");
        }
        return findRunFieldAlias(manifest, normalized);
    }

    private static List<Map<String, Object>> listRunFieldAliases(Map<String, Object> manifest) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("mode", mode);
            row.put("fieldsAlias", "inspectfields:ci");
            row.put("commandAlias", "runcommands:" + mode);
            row.put("allAlias", "runcommands:all");
            copyIfPresent(row, "fieldsCommand", run.get("inspectRunFields"));
            copyIfPresent(row, "command", run.get("inspectRunCommands"));
            copyIfPresent(row, "allCommand", run.get("inspectRunAll"));
            rows.add(row);
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> summarizeRunFields(Map<String, Object> manifest, String summaryMode) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summaryType", "runfields");
        summary.put("summary", summaryMode);
        List<String> modes = new ArrayList<>();
        List<String> fieldAliases = new ArrayList<>();
        List<String> commandAliases = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            modes.add(mode);
            fieldAliases.add("inspectfields:ci");
            commandAliases.add("runcommands:" + mode);
            copyIfPresent(summary, mode + "Fields", run.get("inspectRunFields"));
            copyIfPresent(summary, mode + "Commands", run.get("inspectRunCommands"));
            copyIfPresent(summary, mode + "All", run.get("inspectRunAll"));
        }
        summary.put("modes", List.copyOf(modes));
        summary.put("fieldAliases", List.copyOf(fieldAliases));
        summary.put("commandAliases", List.copyOf(commandAliases));
        return summary;
    }

    private static Map<String, Object> findRunFieldAlias(Map<String, Object> manifest, String mode) {
        for (Map<String, Object> row : listRunFieldAliases(manifest)) {
            if (mode.equals(String.valueOf(row.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT))) {
                return row;
            }
        }
        throw new IllegalArgumentException("Run field mode not found in manifest: " + mode);
    }

    private static Map<String, Object> listInspectFields(Map<String, Object> manifest) {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("manifestFields", List.of(
                "inspectWorkflowStatus",
                "inspectWorkflowHealth",
                "inspectBundleStatus",
                "inspectBundleHealth"));
        List<Map<String, Object>> runFields = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("mode", mode);
            List<String> available = new ArrayList<>();
            addIfPresent(available, run, "inspectRunAll");
            addIfPresent(available, run, "inspectRunCommands");
            addIfPresent(available, run, "inspectWorkflowCommands");
            addIfPresent(available, run, "inspectStatus");
            addIfPresent(available, run, "inspectHealth");
            addIfPresent(available, run, "inspectCi");
            addIfPresent(available, run, "inspectOverview");
            addIfPresent(available, run, "inspectHistory");
            row.put("fields", List.copyOf(available));
            runFields.add(row);
        }
        fields.put("runFields", List.copyOf(runFields));
        return fields;
    }

    private static Object selectInspectFields(Map<String, Object> manifest, String selector) {
        if (selector == null || selector.isBlank()) {
            return listInspectFields(manifest);
        }
        String normalized = selector.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "summary" -> summarizeInspectFields(manifest, "full");
            case "ci" -> summarizeInspectFields(manifest, "short");
            default -> throw new IllegalArgumentException(
                    "Unsupported inspectfields selector '" + selector + "'. Use inspectfields, inspectfields:summary, or inspectfields:ci.");
        };
    }

    private static Map<String, Object> summarizeInspectFields(Map<String, Object> manifest, String summaryMode) {
        Map<String, Object> detail = listInspectFields(manifest);
        List<String> manifestFields = stringList(detail.get("manifestFields"));
        List<Map<String, Object>> runFields = mapList(detail.get("runFields"));
        List<String> runModes = new ArrayList<>();
        List<String> runFieldCounts = new ArrayList<>();
        for (Map<String, Object> row : runFields) {
            String mode = String.valueOf(row.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            List<String> fields = stringList(row.get("fields"));
            runModes.add(mode);
            runFieldCounts.add(mode + ":" + fields.size());
        }

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summaryType", "inspectfields");
        summary.put("summary", summaryMode);
        summary.put("manifestFieldCount", manifestFields.size());
        summary.put("manifestFields", List.copyOf(manifestFields));
        summary.put("runModeCount", runModes.size());
        summary.put("runModes", List.copyOf(runModes));
        summary.put("runFieldCounts", List.copyOf(runFieldCounts));
        return summary;
    }

    private static void addIfPresent(List<String> target, Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value instanceof String text && !text.isBlank()) {
            target.add(key);
        }
    }

    private static List<Map<String, Object>> collectCommandRows(Map<String, Object> manifest, String selector) {
        String expectedScope = null;
        String expectedMode = null;
        String expectedName = null;
        if (selector != null && !selector.isBlank()) {
            for (String rawPart : selector.split(":")) {
                String part = rawPart.trim().toLowerCase(Locale.ROOT);
                if (part.isBlank()) {
                    continue;
                }
                if (part.startsWith("scope=")) {
                    expectedScope = part.substring("scope=".length()).trim();
                    continue;
                }
                if (part.startsWith("mode=")) {
                    expectedMode = part.substring("mode=".length()).trim();
                    continue;
                }
                if (part.startsWith("name=")) {
                    expectedName = part.substring("name=".length()).trim();
                    continue;
                }
                if ("summary".equals(part) || "ci".equals(part)) {
                    continue;
                }
                throw new IllegalArgumentException(
                        "Unsupported commands selector part '" + rawPart + "'. Use summary, ci, scope=workflow|bundle|checkpoint, mode=<run-mode>, or name=<command-name>.");
            }
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        addCommandRow(rows, "workflow", "", "status", manifest.get("inspectWorkflowStatus"));
        addCommandRow(rows, "workflow", "", "health", manifest.get("inspectWorkflowHealth"));
        addCommandRow(rows, "bundle", "", "status", manifest.get("inspectBundleStatus"));
        addCommandRow(rows, "bundle", "", "health", manifest.get("inspectBundleHealth"));
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", ""));
            addCommandRow(rows, "checkpoint", mode, "status", run.get("inspectStatus"));
            addCommandRow(rows, "checkpoint", mode, "health", run.get("inspectHealth"));
            addCommandRow(rows, "checkpoint", mode, "ci", run.get("inspectCi"));
            addCommandRow(rows, "checkpoint", mode, "overview", run.get("inspectOverview"));
            addCommandRow(rows, "checkpoint", mode, "history", run.get("inspectHistory"));
        }
        if (expectedScope != null) {
            rows.removeIf(row -> !expectedScope.equals(String.valueOf(row.getOrDefault("scope", "")).toLowerCase(Locale.ROOT)));
        }
        if (expectedMode != null) {
            rows.removeIf(row -> !expectedMode.equals(String.valueOf(row.getOrDefault("mode", "")).toLowerCase(Locale.ROOT)));
        }
        if (expectedName != null) {
            rows.removeIf(row -> !expectedName.equals(String.valueOf(row.getOrDefault("name", "")).toLowerCase(Locale.ROOT)));
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> summarizeCommands(
            Map<String, Object> manifest,
            String expectedScope,
            String expectedMode,
            String expectedName,
            String summaryMode) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summaryType", "commands");
        summary.put("summary", summaryMode);
        copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "workflow", "", "status", "workflowStatus", manifest.get("inspectWorkflowStatus"));
        copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "workflow", "", "health", "workflowHealth", manifest.get("inspectWorkflowHealth"));
        copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "bundle", "", "status", "bundleStatus", manifest.get("inspectBundleStatus"));
        copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "bundle", "", "health", "bundleHealth", manifest.get("inspectBundleHealth"));
        for (Map<String, Object> run : runs(manifest)) {
            String mode = String.valueOf(run.getOrDefault("mode", "")).trim().toLowerCase(Locale.ROOT);
            if (mode.isBlank()) {
                continue;
            }
            copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "checkpoint", mode, "status", mode + "Status", run.get("inspectStatus"));
            copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "checkpoint", mode, "health", mode + "Health", run.get("inspectHealth"));
            copyCommandIfAccepted(summary, expectedScope, expectedMode, expectedName, "checkpoint", mode, "ci", mode + "Ci", run.get("inspectCi"));
        }
        return summary;
    }

    private static void copyCommandIfAccepted(
            Map<String, Object> target,
            String expectedScope,
            String expectedMode,
            String expectedName,
            String scope,
            String mode,
            String name,
            String key,
            Object value) {
        if (expectedScope != null && !expectedScope.equals(scope)) {
            return;
        }
        if (expectedMode != null && !expectedMode.equals(mode)) {
            return;
        }
        if (expectedName != null && !expectedName.equals(name)) {
            return;
        }
        copyIfPresent(target, key, value);
    }

    private static void copyIfPresent(Map<String, Object> target, String key, Object value) {
        if (value instanceof String text && !text.isBlank()) {
            target.put(key, text);
        }
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<String> rows = new ArrayList<>();
        for (Object item : list) {
            rows.add(String.valueOf(item));
        }
        return List.copyOf(rows);
    }

    private static List<Map<String, Object>> mapList(Object value) {
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof Map<?, ?> map) {
                Map<String, Object> copy = new LinkedHashMap<>();
                map.forEach((key, element) -> copy.put(String.valueOf(key), element));
                rows.add(copy);
            }
        }
        return List.copyOf(rows);
    }

    private static void addCommandRow(List<Map<String, Object>> rows, String scope, String mode, String name, Object commandValue) {
        if (!(commandValue instanceof String command) || command.isBlank()) {
            return;
        }
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("scope", scope);
        row.put("mode", mode);
        row.put("name", name);
        row.put("command", command);
        rows.add(row);
    }

    private static Map<String, Object> findRun(Map<String, Object> manifest, String mode) {
        for (Map<String, Object> run : runs(manifest)) {
            if (mode.equalsIgnoreCase(String.valueOf(run.getOrDefault("mode", "")))) {
                return run;
            }
        }
        throw new IllegalArgumentException("Run mode not found in manifest: " + mode);
    }

    private static List<Map<String, Object>> collectMetricRows(Map<String, Object> manifest, String key) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map<String, Object> run : runs(manifest)) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("mode", run.getOrDefault("mode", ""));
            row.put(key, run.getOrDefault(key, ""));
            rows.add(row);
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> computeDelta(Map<String, Object> manifest, String key) {
        Map<String, Object> fresh = maybeFindRun(manifest, "fresh");
        Map<String, Object> resume = maybeFindRun(manifest, "resume");
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("metric", key);
        delta.put("fresh", fresh.getOrDefault(key, ""));
        delta.put("resume", resume.getOrDefault(key, ""));
        Double freshValue = parseDouble(fresh.get(key));
        Double resumeValue = parseDouble(resume.get(key));
        if (freshValue != null && resumeValue != null) {
            delta.put("difference", resumeValue - freshValue);
        }
        return delta;
    }

    private static Map<String, Object> maybeFindRun(Map<String, Object> manifest, String mode) {
        for (Map<String, Object> run : runs(manifest)) {
            if (mode.equalsIgnoreCase(String.valueOf(run.getOrDefault("mode", "")))) {
                return run;
            }
        }
        return Map.of();
    }

    private static Double parseDouble(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value);
        if (text.isBlank() || "unknown".equalsIgnoreCase(text)) {
            return null;
        }
        try {
            return Double.parseDouble(text);
        } catch (NumberFormatException exception) {
            return null;
        }
    }

    private static List<Map<String, Object>> runs(Map<String, Object> manifest) {
        Object value = manifest.get("runs");
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof Map<?, ?> row) {
                Map<String, Object> copy = new LinkedHashMap<>();
                row.forEach((key, element) -> copy.put(String.valueOf(key), element));
                rows.add(copy);
            }
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> loadJsonMap(Path file) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> values = JSON.readValue(file.toFile(), LinkedHashMap.class);
            return values;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load JSON file " + file + ".", exception);
        }
    }

    private static String normalizeFormat(String format, Path outputPath) {
        if (format != null && !format.isBlank()) {
            return format.toLowerCase(Locale.ROOT);
        }
        if (outputPath != null) {
            String name = outputPath.getFileName().toString().toLowerCase(Locale.ROOT);
            if (name.endsWith(".json")) {
                return "json";
            }
            if (name.endsWith(".csv")) {
                return "csv";
            }
        }
        return "text";
    }

    private static String renderValue(Object value, String format) {
        return switch (format) {
            case "json" -> renderJson(value);
            case "csv" -> renderCsv(value);
            case "text" -> renderText(value);
            default -> throw new IllegalArgumentException("Unsupported format '" + format + "'. Use text, json, or csv.");
        };
    }

    private static String renderText(Object value) {
        if (value instanceof Map<?, ?> map) {
            if (isCommandSummaryMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderCommandSummaryText(typedMap);
            }
            if (isRunCommandSummaryMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderRunCommandSummaryText(typedMap);
            }
            if (isRunFieldSummaryMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderRunFieldSummaryText(typedMap);
            }
            if (isInspectFieldSummaryMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderInspectFieldSummaryText(typedMap);
            }
            if (isBundleOverviewMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderBundleOverviewText(typedMap);
            }
            if (isBundleHealthMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderBundleHealthText(typedMap);
            }
            StringBuilder out = new StringBuilder();
            map.forEach((key, element) -> out.append(key).append('=').append(String.valueOf(element)).append('\n'));
            return out.toString();
        }
        if (value instanceof List<?> list) {
            if (list.isEmpty()) {
                return "(empty)\n";
            }
            if (list.get(0) instanceof Map<?, ?>) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rows = (List<Map<String, Object>>) list;
                return renderTextTable(rows);
            }
            StringBuilder out = new StringBuilder();
            for (Object item : list) {
                out.append(String.valueOf(item)).append('\n');
            }
            return out.toString();
        }
        return String.valueOf(value) + '\n';
    }

    private static boolean isBundleOverviewMap(Map<?, ?> map) {
        return map.containsKey("workflowBundleDir")
                && map.containsKey("workflowBundleStatus")
                && map.containsKey("bundleAvailable")
                && map.containsKey("focus");
    }

    private static boolean isCommandSummaryMap(Map<?, ?> map) {
        return "commands".equals(String.valueOf(map.getOrDefault("summaryType", "")))
                && map.containsKey("summary");
    }

    private static boolean isRunCommandSummaryMap(Map<?, ?> map) {
        return "runcommands".equals(String.valueOf(map.getOrDefault("summaryType", "")))
                && map.containsKey("summary");
    }

    private static boolean isInspectFieldSummaryMap(Map<?, ?> map) {
        return "inspectfields".equals(String.valueOf(map.getOrDefault("summaryType", "")))
                && map.containsKey("summary");
    }

    private static boolean isRunFieldSummaryMap(Map<?, ?> map) {
        return "runfields".equals(String.valueOf(map.getOrDefault("summaryType", "")))
                && map.containsKey("summary");
    }

    private static boolean isBundleHealthMap(Map<?, ?> map) {
        return map.containsKey("bundleType")
                && map.containsKey("outputDirectory")
                && map.containsKey("expectedFileCount")
                && map.containsKey("missingFileCount")
                && map.containsKey("status")
                && map.containsKey("summary");
    }

    private static String renderBundleOverviewText(Map<String, Object> map) {
        String bundleDir = String.valueOf(map.getOrDefault("workflowBundleDir", ""));
        String bundleStatus = String.valueOf(map.getOrDefault("workflowBundleStatus", ""));
        String focus = String.valueOf(map.getOrDefault("focus", "all"));
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        boolean available = Boolean.parseBoolean(String.valueOf(map.getOrDefault("bundleAvailable", false)));
        if ("short".equals(summary)) {
            StringBuilder shortOut = new StringBuilder();
            shortOut.append("bundleOverview")
                    .append(" status=").append(bundleStatus)
                    .append(" available=").append(available)
                    .append(" focus=").append(focus);
            if (map.containsKey("bundleType")) {
                shortOut.append(" type=").append(String.valueOf(map.getOrDefault("bundleType", "")))
                        .append(" count=").append(String.valueOf(map.getOrDefault("fileCount", 0)));
            }
            if (map.containsKey("healthStatus")) {
                shortOut.append(" health=").append(String.valueOf(map.getOrDefault("healthStatus", "")))
                        .append(" missing=").append(String.valueOf(map.getOrDefault("missingFileCount", 0)));
            }
            shortOut.append(" dir=").append(bundleDir);
            return shortOut.append('\n').toString();
        }
        StringBuilder out = new StringBuilder();
        out.append("bundleOverview")
                .append(" status=").append(bundleStatus)
                .append(" available=").append(available)
                .append(" focus=").append(focus)
                .append('\n');
        out.append("dir=").append(bundleDir).append('\n');
        if (map.containsKey("bundleType")) {
            out.append("files")
                    .append(" type=").append(String.valueOf(map.getOrDefault("bundleType", "")))
                    .append(" count=").append(String.valueOf(map.getOrDefault("fileCount", 0)))
                    .append('\n');
        }
        if (map.containsKey("healthStatus")) {
            out.append("health")
                    .append(" status=").append(String.valueOf(map.getOrDefault("healthStatus", "")))
                    .append(" missing=").append(String.valueOf(map.getOrDefault("missingFileCount", 0)))
                    .append('\n');
        }
        if (map.containsKey("summaryMessage")) {
            out.append("summary=").append(String.valueOf(map.getOrDefault("summaryMessage", ""))).append('\n');
        }
        return out.toString();
    }

    private static String renderCommandSummaryText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        List<String> entries = new ArrayList<>();
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            String key = entry.getKey();
            if ("summaryType".equals(key) || "summary".equals(key)) {
                continue;
            }
            entries.add(key + "=" + String.valueOf(entry.getValue()));
        }
        if ("short".equals(summary)) {
            return "commands " + String.join(" ", entries) + '\n';
        }
        StringBuilder out = new StringBuilder();
        out.append("commands")
                .append(" summary=").append(summary)
                .append('\n');
        for (String entry : entries) {
            out.append(entry).append('\n');
        }
        return out.toString();
    }

    private static String renderRunCommandSummaryText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        List<String> modes = new ArrayList<>();
        Object modesValue = map.get("modes");
        if (modesValue instanceof List<?> list) {
            for (Object item : list) {
                modes.add(String.valueOf(item));
            }
        }
        List<String> aliases = new ArrayList<>();
        Object aliasesValue = map.get("aliases");
        if (aliasesValue instanceof List<?> list) {
            for (Object item : list) {
                aliases.add(String.valueOf(item));
            }
        }
        if ("short".equals(summary)) {
            StringBuilder out = new StringBuilder();
            out.append("runcommands");
            if (!modes.isEmpty()) {
                out.append(" modes=").append(String.join(",", modes));
            }
            if (!aliases.isEmpty()) {
                out.append(" aliases=").append(String.join(",", aliases));
            }
            return out.append('\n').toString();
        }
        StringBuilder out = new StringBuilder();
        out.append("runcommands")
                .append(" summary=").append(summary)
                .append('\n');
        if (!modes.isEmpty()) {
            out.append("modes=").append(modes).append('\n');
        }
        if (!aliases.isEmpty()) {
            out.append("aliases=").append(aliases).append('\n');
        }
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            String key = entry.getKey();
            if ("summaryType".equals(key) || "summary".equals(key) || "modes".equals(key) || "aliases".equals(key)) {
                continue;
            }
            out.append(key).append('=').append(String.valueOf(entry.getValue())).append('\n');
        }
        return out.toString();
    }

    private static String renderInspectFieldSummaryText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        List<String> manifestFields = stringList(map.get("manifestFields"));
        List<String> runModes = stringList(map.get("runModes"));
        List<String> runFieldCounts = stringList(map.get("runFieldCounts"));
        if ("short".equals(summary)) {
            StringBuilder out = new StringBuilder();
            out.append("inspectfields")
                    .append(" manifestFields=").append(String.valueOf(map.getOrDefault("manifestFieldCount", 0)))
                    .append(" runModes=").append(String.join(",", runModes))
                    .append(" runFieldCounts=").append(String.join(",", runFieldCounts));
            return out.append('\n').toString();
        }
        StringBuilder out = new StringBuilder();
        out.append("inspectfields")
                .append(" summary=").append(summary)
                .append('\n');
        out.append("manifestFieldCount=").append(String.valueOf(map.getOrDefault("manifestFieldCount", 0))).append('\n');
        out.append("manifestFields=").append(manifestFields).append('\n');
        out.append("runModeCount=").append(String.valueOf(map.getOrDefault("runModeCount", 0))).append('\n');
        out.append("runModes=").append(runModes).append('\n');
        out.append("runFieldCounts=").append(runFieldCounts).append('\n');
        return out.toString();
    }

    private static String renderRunFieldSummaryText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        List<String> modes = stringList(map.get("modes"));
        List<String> fieldAliases = stringList(map.get("fieldAliases"));
        List<String> commandAliases = stringList(map.get("commandAliases"));
        if ("short".equals(summary)) {
            StringBuilder out = new StringBuilder();
            out.append("runfields");
            if (!modes.isEmpty()) {
                out.append(" modes=").append(String.join(",", modes));
            }
            if (!fieldAliases.isEmpty()) {
                out.append(" fieldAliases=").append(String.join(",", fieldAliases));
            }
            if (!commandAliases.isEmpty()) {
                out.append(" commandAliases=").append(String.join(",", commandAliases));
            }
            return out.append('\n').toString();
        }
        StringBuilder out = new StringBuilder();
        out.append("runfields")
                .append(" summary=").append(summary)
                .append('\n');
        if (!modes.isEmpty()) {
            out.append("modes=").append(modes).append('\n');
        }
        if (!fieldAliases.isEmpty()) {
            out.append("fieldAliases=").append(fieldAliases).append('\n');
        }
        if (!commandAliases.isEmpty()) {
            out.append("commandAliases=").append(commandAliases).append('\n');
        }
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            String key = entry.getKey();
            if ("summaryType".equals(key)
                    || "summary".equals(key)
                    || "modes".equals(key)
                    || "fieldAliases".equals(key)
                    || "commandAliases".equals(key)) {
                continue;
            }
            out.append(key).append('=').append(String.valueOf(entry.getValue())).append('\n');
        }
        return out.toString();
    }

    private static String renderBundleHealthText(Map<String, Object> map) {
        String bundleType = String.valueOf(map.getOrDefault("bundleType", ""));
        String outputDirectory = String.valueOf(map.getOrDefault("outputDirectory", ""));
        String status = String.valueOf(map.getOrDefault("status", ""));
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        String expectedFileCount = String.valueOf(map.getOrDefault("expectedFileCount", 0));
        String missingFileCount = String.valueOf(map.getOrDefault("missingFileCount", 0));
        if ("short".equals(summary)) {
            return new StringBuilder()
                    .append("bundleHealth")
                    .append(" status=").append(status)
                    .append(" type=").append(bundleType)
                    .append(" expected=").append(expectedFileCount)
                    .append(" missing=").append(missingFileCount)
                    .append(" dir=").append(outputDirectory)
                    .append('\n')
                    .toString();
        }
        StringBuilder out = new StringBuilder();
        out.append("bundleHealth")
                .append(" status=").append(status)
                .append(" type=").append(bundleType)
                .append(" summary=").append(summary)
                .append('\n');
        out.append("dir=").append(outputDirectory).append('\n');
        out.append("files expected=").append(expectedFileCount)
                .append(" missing=").append(missingFileCount)
                .append('\n');
        out.append("summary=").append(String.valueOf(map.getOrDefault("summaryMessage", ""))).append('\n');
        Object missingFiles = map.get("missingFiles");
        if (missingFiles instanceof List<?> list && !list.isEmpty()) {
            out.append("missingFiles=").append(list).append('\n');
        }
        return out.toString();
    }

    private static String renderJson(Object value) {
        try {
            return JSON.writeValueAsString(value) + '\n';
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render JSON output.", exception);
        }
    }

    private static String renderCsv(Object value) {
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> row = (Map<String, Object>) map;
            return renderCsvRows(List.of(row));
        }
        if (value instanceof List<?> list) {
            if (list.isEmpty()) {
                return "";
            }
            if (!(list.get(0) instanceof Map<?, ?>)) {
                throw new IllegalArgumentException("CSV output requires a map or list of maps.");
            }
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rows = (List<Map<String, Object>>) list;
            return renderCsvRows(rows);
        }
        throw new IllegalArgumentException("CSV output requires a map or list of maps.");
    }

    private static String renderTextTable(List<Map<String, Object>> rows) {
        List<String> headers = collectHeaders(rows);
        List<Integer> widths = new ArrayList<>();
        for (String header : headers) {
            int width = header.length();
            for (Map<String, Object> row : rows) {
                width = Math.max(width, String.valueOf(row.getOrDefault(header, "")).length());
            }
            widths.add(width);
        }
        StringBuilder out = new StringBuilder();
        appendTableRow(out, headers, widths);
        appendSeparator(out, widths);
        for (Map<String, Object> row : rows) {
            List<String> cells = headers.stream()
                    .map(header -> String.valueOf(row.getOrDefault(header, "")))
                    .toList();
            appendTableRow(out, cells, widths);
        }
        return out.toString();
    }

    private static String renderCsvRows(List<Map<String, Object>> rows) {
        List<String> headers = collectHeaders(rows);
        StringBuilder out = new StringBuilder();
        out.append(String.join(",", headers.stream().map(trainer_byte_latent_workflow_inspector::escapeCsv).toList()))
                .append('\n');
        for (Map<String, Object> row : rows) {
            out.append(String.join(
                            ",",
                            headers.stream()
                                    .map(header -> escapeCsv(String.valueOf(row.getOrDefault(header, ""))))
                                    .toList()))
                    .append('\n');
        }
        return out.toString();
    }

    private static List<String> collectHeaders(List<Map<String, Object>> rows) {
        LinkedHashSet<String> headers = new LinkedHashSet<>();
        for (Map<String, Object> row : rows) {
            headers.addAll(row.keySet());
        }
        return List.copyOf(headers);
    }

    private static void appendTableRow(StringBuilder out, List<String> cells, List<Integer> widths) {
        for (int index = 0; index < cells.size(); index++) {
            if (index > 0) {
                out.append(" | ");
            }
            out.append(padRight(cells.get(index), widths.get(index)));
        }
        out.append('\n');
    }

    private static void appendSeparator(StringBuilder out, List<Integer> widths) {
        for (int index = 0; index < widths.size(); index++) {
            if (index > 0) {
                out.append("-+-");
            }
            out.append("-".repeat(widths.get(index)));
        }
        out.append('\n');
    }

    private static String padRight(String value, int width) {
        return String.format("%-" + width + "s", value);
    }

    private static String escapeCsv(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private static void writeOutput(Path outputPath, String rendered) {
        try {
            Path parent = outputPath.toAbsolutePath().normalize().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(outputPath, rendered);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write output file " + outputPath + ".", exception);
        }
    }

    private static CliOptions parseArgs(String[] args) {
        String section = args.length > 1 ? args[1] : "overview";
        String format = args.length > 2 ? args[2] : null;
        Path outputPath = args.length > 3 ? Path.of(args[3]) : null;
        return new CliOptions(section, format, outputPath);
    }

    private record BundleOverviewOptions(String focus, String summary) {}
    private record CliOptions(String section, String format, Path outputPath) {}
}
