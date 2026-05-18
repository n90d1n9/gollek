///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS com.fasterxml.jackson.core:jackson-databind:2.20.1

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * JBang inspector for byte-latent train+infer report artifacts.
 */
public class trainer_byte_latent_train_infer_inspector {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String REPORT_FILE_NAME = "train-infer-report.json";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the checkpoint directory or train-infer-report.json path as the first argument.");
        }
        CliOptions options = parseArgs(args);
        Path inputPath = Path.of(args[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        InputBundle input = resolveInput(inputPath);
        Object value = select(input, section);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Train+Infer Inspector");
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
                    "Train+infer report not found: " + reportFile.toAbsolutePath().normalize());
        }
        return new InputBundle(inputPath, reportFile, loadJsonMap(reportFile));
    }

    private static Object select(InputBundle input, String section) {
        String normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        if ("all".equals(normalized) || "report".equals(normalized)) {
            return input.report();
        }
        if ("overview".equals(normalized)) {
            return summarizeOverview(input);
        }
        if ("status".equals(normalized)) {
            return summarizeStatus(input, false);
        }
        if ("ci".equals(normalized) || "status:ci".equals(normalized)) {
            return summarizeStatus(input, true);
        }
        if ("training".equals(normalized)) {
            return asMap(input.report().get("training"));
        }
        if ("inference".equals(normalized)) {
            return asMap(input.report().get("inference"));
        }
        if ("nexttoken".equals(normalized)) {
            return selectObjectPath(input.report().get("inference"), "nextToken");
        }
        if ("generatedtext".equals(normalized)) {
            return selectObjectPath(input.report().get("inference"), "generatedText");
        }
        if ("combinedtext".equals(normalized)) {
            return selectObjectPath(input.report().get("inference"), "combinedText");
        }
        if ("prompt".equals(normalized)) {
            return input.report().getOrDefault("prompt", "");
        }
        if ("maxnewtokens".equals(normalized)) {
            return input.report().getOrDefault("maxNewTokens", "");
        }
        if (normalized.startsWith("training:")) {
            return selectObjectPath(input.report().get("training"), section.substring("training:".length()));
        }
        if (normalized.startsWith("inference:")) {
            return selectObjectPath(input.report().get("inference"), section.substring("inference:".length()));
        }
        throw new IllegalArgumentException(
                "Unknown section '" + section + "'. Use one of: overview, status, ci, status:ci, report, training, inference, nextToken, generatedText, combinedText, prompt, maxNewTokens, training:<path>, or inference:<path>.");
    }

    private static Map<String, Object> summarizeOverview(InputBundle input) {
        Map<String, Object> report = input.report();
        Map<String, Object> training = asMap(report.get("training"));
        Map<String, Object> inference = asMap(report.get("inference"));
        Map<String, Object> metadata = asMap(training.get("metadata"));
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("familyId", report.getOrDefault("familyId", ""));
        overview.put("prompt", report.getOrDefault("prompt", ""));
        overview.put("maxNewTokens", report.getOrDefault("maxNewTokens", ""));
        overview.put("epochCount", training.getOrDefault("epochCount", ""));
        overview.put("latestTrainLoss", training.getOrDefault("latestTrainLoss", ""));
        overview.put("bestValidationLoss", training.getOrDefault("bestValidationLoss", ""));
        overview.put("globalStep", metadata.getOrDefault("globalStep", ""));
        overview.put("nextToken", inference.getOrDefault("nextToken", ""));
        overview.put("generatedText", inference.getOrDefault("generatedText", ""));
        overview.put("combinedText", inference.getOrDefault("combinedText", ""));
        overview.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        return overview;
    }

    private static Map<String, Object> summarizeStatus(InputBundle input, boolean ci) {
        Map<String, Object> report = input.report();
        Map<String, Object> training = asMap(report.get("training"));
        Map<String, Object> inference = asMap(report.get("inference"));
        Map<String, Object> metadata = asMap(training.get("metadata"));
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", ci ? "short" : "full");
        status.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        status.put("checkpointDir", report.getOrDefault("checkpointDir", ""));
        status.put("prompt", report.getOrDefault("prompt", ""));
        status.put("epochCount", training.getOrDefault("epochCount", ""));
        status.put("latestTrainLoss", training.getOrDefault("latestTrainLoss", ""));
        status.put("globalStep", metadata.getOrDefault("globalStep", ""));
        status.put("nextToken", inference.getOrDefault("nextToken", ""));
        status.put("combinedText", inference.getOrDefault("combinedText", ""));
        return status;
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

    private static Object selectObjectPath(Object root, String path) {
        Object current = root;
        for (String segment : path.split(":")) {
            if (segment.isBlank()) {
                continue;
            }
            if (current instanceof Map<?, ?> map) {
                current = map.get(segment);
                continue;
            }
            throw new IllegalArgumentException("Cannot descend into path segment '" + segment + "'.");
        }
        return current;
    }

    private static Map<String, Object> asMap(Object value) {
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        map.forEach((key, element) -> copy.put(String.valueOf(key), element));
        return copy;
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
            if (isStatusMap(map)) {
                @SuppressWarnings("unchecked")
                Map<String, Object> typedMap = (Map<String, Object>) map;
                return renderStatusText(typedMap);
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

    private static boolean isStatusMap(Map<?, ?> map) {
        return map.containsKey("summary")
                && map.containsKey("reportFile")
                && map.containsKey("checkpointDir");
    }

    private static String renderStatusText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        if ("short".equals(summary)) {
            return "status"
                    + " epochCount=" + map.getOrDefault("epochCount", "")
                    + " latestTrainLoss=" + map.getOrDefault("latestTrainLoss", "")
                    + " globalStep=" + map.getOrDefault("globalStep", "")
                    + " nextToken=" + map.getOrDefault("nextToken", "")
                    + " combinedText=" + map.getOrDefault("combinedText", "")
                    + " reportFile=" + map.getOrDefault("reportFile", "")
                    + '\n';
        }
        StringBuilder out = new StringBuilder();
        map.forEach((key, value) -> out.append(key).append('=').append(String.valueOf(value)).append('\n'));
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
        out.append(String.join(",", headers.stream().map(trainer_byte_latent_train_infer_inspector::escapeCsv).toList()))
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

    private static String padRight(String text, int width) {
        return text + " ".repeat(Math.max(0, width - text.length()));
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
            Files.writeString(outputPath, rendered, StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write output to " + outputPath + ".", exception);
        }
    }

    private static CliOptions parseArgs(String[] args) {
        String section = args.length >= 2 ? args[1] : "overview";
        String format = args.length >= 3 ? args[2] : null;
        Path outputPath = args.length >= 4 ? Path.of(args[3]) : null;
        return new CliOptions(section, format, outputPath);
    }

    private record CliOptions(String section, String format, Path outputPath) {
    }

    private record InputBundle(Path inputPath, Path reportFile, Map<String, Object> report) {
    }
}
