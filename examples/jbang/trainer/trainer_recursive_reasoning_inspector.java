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
 * JBang inspector for recursive-reasoning report artifacts.
 */
public class trainer_recursive_reasoning_inspector {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String REPORT_FILE_NAME = "recursive-reasoning-report.json";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the output directory or recursive-reasoning-report.json path as the first argument.");
        }
        CliOptions options = parseArgs(args);
        Path inputPath = Path.of(args[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        InputBundle input = resolveInput(inputPath);
        Object value = select(input, section);

        System.out.println("====================================================");
        System.out.println(" Gollek Recursive Reasoning Inspector");
        System.out.println("====================================================");
        System.out.println("inputPath=" + inputPath.toAbsolutePath().normalize());
        System.out.println("reportFile=" + input.reportFile().toAbsolutePath().normalize());
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

    private static Object select(InputBundle input, String section) {
        String normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        if ("all".equals(normalized) || "report".equals(normalized)) {
            return input.report();
        }
        if ("reportversion".equals(normalized)) {
            return input.report().getOrDefault("reportVersion", "1");
        }
        if ("overview".equals(normalized)) {
            return summarizeOverview(input, false);
        }
        if ("overview:ci".equals(normalized)) {
            return summarizeOverview(input, true);
        }
        if ("status".equals(normalized) || "health".equals(normalized)) {
            return summarizeStatus(input, false);
        }
        if ("ci".equals(normalized) || "status:ci".equals(normalized) || "health:ci".equals(normalized)) {
            return summarizeStatus(input, true);
        }
        if ("summary".equals(normalized)) {
            return asMap(input.report().get("summary"));
        }
        if ("config".equals(normalized)) {
            return asMap(input.report().get("config"));
        }
        if ("taskid".equals(normalized)) {
            return input.report().getOrDefault("taskId", "");
        }
        if ("familyid".equals(normalized)) {
            return input.report().getOrDefault("familyId", "");
        }
        if ("selectedstateid".equals(normalized)) {
            return selectedStateId(input.report(), asMap(input.report().get("summary")));
        }
        if ("selectedtrajectoryindex".equals(normalized)) {
            return input.report().getOrDefault("selectedTrajectoryIndex", "");
        }
        if ("selectedrewardscore".equals(normalized)) {
            return input.report().getOrDefault("selectedRewardScore", "");
        }
        if ("selectedcumulativelogprobability".equals(normalized)) {
            return input.report().getOrDefault("selectedCumulativeLogProbability", "");
        }
        if (normalized.startsWith("config:")) {
            return selectObjectPath(input.report().get("config"), section.substring("config:".length()));
        }
        if (normalized.startsWith("summary:")) {
            return selectObjectPath(input.report().get("summary"), section.substring("summary:".length()));
        }
        throw new IllegalArgumentException(
                "Unknown section '" + section + "'. Use one of: reportVersion, overview, overview:ci, status, health, ci, summary, config, taskId, familyId, selectedStateId, selectedTrajectoryIndex, selectedRewardScore, selectedCumulativeLogProbability, report, config:<path>, or summary:<path>.");
    }

    private static Map<String, Object> summarizeOverview(InputBundle input, boolean ci) {
        Map<String, Object> report = input.report();
        Map<String, Object> summary = asMap(report.get("summary"));
        if (ci) {
            Map<String, Object> compact = new LinkedHashMap<>();
            compact.put("summary", "short");
            compact.put("familyId", report.getOrDefault("familyId", ""));
            compact.put("taskId", report.getOrDefault("taskId", ""));
            compact.put("selectedStateId", selectedStateId(report, summary));
            compact.put("selectedTrajectoryIndex", report.getOrDefault("selectedTrajectoryIndex", ""));
            compact.put("selectedRewardScore", report.getOrDefault("selectedRewardScore", ""));
            compact.put("selectedCumulativeLogProbability", report.getOrDefault("selectedCumulativeLogProbability", ""));
            compact.put("exploredTrajectoryCount", summary.getOrDefault("exploredTrajectoryCount", ""));
            compact.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
            return compact;
        }
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("reportVersion", report.getOrDefault("reportVersion", "1"));
        overview.put("familyId", report.getOrDefault("familyId", ""));
        overview.put("taskId", report.getOrDefault("taskId", ""));
        overview.put("config", asMap(report.get("config")));
        overview.put("summary", summary);
        overview.put("selectedTrajectoryIndex", report.getOrDefault("selectedTrajectoryIndex", ""));
        overview.put("selectedStateId", selectedStateId(report, summary));
        overview.put("selectedRewardScore", report.getOrDefault("selectedRewardScore", ""));
        overview.put("selectedCumulativeLogProbability", report.getOrDefault("selectedCumulativeLogProbability", ""));
        overview.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        return overview;
    }

    private static Map<String, Object> summarizeStatus(InputBundle input, boolean ci) {
        Map<String, Object> report = input.report();
        Map<String, Object> summary = asMap(report.get("summary"));
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", ci ? "short" : "full");
        status.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        status.put("taskId", report.getOrDefault("taskId", ""));
        status.put("selectedStateId", selectedStateId(report, summary));
        status.put("selectedTrajectoryIndex", report.getOrDefault("selectedTrajectoryIndex", ""));
        status.put("selectedRewardScore", report.getOrDefault("selectedRewardScore", ""));
        status.put("exploredTrajectoryCount", summary.getOrDefault("exploredTrajectoryCount", ""));
        status.put("completedTrajectoryCount", summary.getOrDefault("completedTrajectoryCount", ""));
        return status;
    }

    private static Object selectedStateId(Map<String, Object> report, Map<String, Object> summary) {
        Object topLevel = report.get("selectedStateId");
        if (topLevel != null && !String.valueOf(topLevel).isBlank()) {
            return topLevel;
        }
        return summary.getOrDefault("selectedStateId", "");
    }

    private static Object selectObjectPath(Object root, String path) {
        Object current = root;
        for (String segment : path.split(":")) {
            if (!(current instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException("Cannot descend into '" + segment + "' for path '" + path + "'.");
            }
            current = map.get(segment);
        }
        return current;
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

    private static CliOptions parseArgs(String[] args) {
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
        return new CliOptions(section, format, outputPath);
    }

    private static boolean isFormatToken(String token) {
        String normalized = token.toLowerCase(Locale.ROOT);
        return "json".equals(normalized) || "text".equals(normalized);
    }

    private record InputBundle(Path inputPath, Path reportFile, Map<String, Object> report) {
    }

    private record CliOptions(String section, String format, Path outputPath) {
    }
}
