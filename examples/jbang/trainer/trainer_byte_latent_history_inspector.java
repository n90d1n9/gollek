///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-byte-latent:0.1.0-SNAPSHOT
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
import tech.kayys.gollek.ml.bytelatent.ByteLatentHistoryReports;
import tech.kayys.gollek.ml.bytelatent.ByteLatentHistoryRow;

/**
 * JBang inspector for byte-latent checkpoint summary/history artifacts.
 */
public class trainer_byte_latent_history_inspector {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String HISTORY_FILE_NAME = "byte-latent-history.csv";
    private static final String REPORT_FILE_NAME = "byte-latent-report.json";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException(
                    "Pass the checkpoint directory or byte-latent-history.csv path as the first argument.");
        }
        CliOptions options = parseArgs(args);
        Path inputPath = Path.of(args[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        InputBundle input = resolveInput(inputPath);
        Object value = select(input, section);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent History Inspector");
        System.out.println("====================================================");
        System.out.println("inputPath=" + inputPath.toAbsolutePath().normalize());
        System.out.println("reportFile=" + String.valueOf(input.reportFile()));
        System.out.println("historyFile=" + input.historyFile().toAbsolutePath().normalize());
        System.out.println("section=" + (section == null ? "all" : section));
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
        if (Files.isDirectory(inputPath)) {
            Path historyFile = inputPath.resolve(HISTORY_FILE_NAME);
            Path reportFile = inputPath.resolve(REPORT_FILE_NAME);
            if (!Files.isRegularFile(historyFile)) {
                throw new IllegalArgumentException("Byte-latent history file not found: " + historyFile.toAbsolutePath().normalize());
            }
            Map<String, Object> report = Files.isRegularFile(reportFile) ? loadJsonMap(reportFile) : Map.of();
            return new InputBundle(inputPath, historyFile, Files.isRegularFile(reportFile) ? reportFile : null, report);
        }
        String name = inputPath.getFileName() == null ? "" : inputPath.getFileName().toString();
        if (REPORT_FILE_NAME.equalsIgnoreCase(name)) {
            Map<String, Object> report = loadJsonMap(inputPath);
            Object artifactsValue = report.get("artifacts");
            Path historyFile = inputPath.resolveSibling(HISTORY_FILE_NAME);
            if (artifactsValue instanceof Map<?, ?> artifacts
                    && artifacts.get("historyFile") instanceof String historyPath) {
                historyFile = Path.of(historyPath);
            }
            if (!Files.isRegularFile(historyFile)) {
                throw new IllegalArgumentException("Byte-latent history file not found: " + historyFile.toAbsolutePath().normalize());
            }
            return new InputBundle(inputPath, historyFile, inputPath, report);
        }
        if (!Files.isRegularFile(inputPath)) {
            throw new IllegalArgumentException("Byte-latent history file not found: " + inputPath.toAbsolutePath().normalize());
        }
        return new InputBundle(inputPath, inputPath, null, Map.of());
    }

    private static Object select(InputBundle input, String section) {
        String normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        if ("status".equals(normalized)) {
            return summarizeStatus(input, false);
        }
        if ("health".equals(normalized)) {
            return summarizeStatus(input, true);
        }
        if ("status:ci".equals(normalized) || "ci".equals(normalized)) {
            return summarizeStatus(input, true);
        }
        if ("health:ci".equals(normalized)) {
            return summarizeStatus(input, true);
        }
        if (input.reportFile() == null) {
            return ByteLatentHistoryReports.select(input.historyFile(), normalized.startsWith("history") ? normalized : "history:" + normalized);
        }
        if ("all".equals(normalized) || "overview".equals(normalized)) {
            return Map.of(
                    "summary", input.report().getOrDefault("summary", Map.of()),
                    "historyCount", input.report().getOrDefault("historyCount", 0),
                    "artifacts", input.report().getOrDefault("artifacts", Map.of()));
        }
        if ("report".equals(normalized)) {
            return input.report();
        }
        if ("summary".equals(normalized)) {
            return input.report().getOrDefault("summary", Map.of());
        }
        if ("artifacts".equals(normalized)) {
            return input.report().getOrDefault("artifacts", Map.of());
        }
        if ("historycount".equals(normalized)) {
            return input.report().getOrDefault("historyCount", 0);
        }
        if (normalized.startsWith("summary:")) {
            return selectObjectPath(input.report().get("summary"), normalized.substring("summary:".length()));
        }
        if (normalized.startsWith("artifacts:")) {
            return selectObjectPath(input.report().get("artifacts"), normalized.substring("artifacts:".length()));
        }
        if (normalized.startsWith("history:") || "history".equals(normalized)) {
            return ByteLatentHistoryReports.select(historyRows(input.report()), normalized);
        }
        throw new IllegalArgumentException(
                "Unknown section '" + section + "'. Use one of: overview, status, health, ci, status:ci, health:ci, report, summary, artifacts, historyCount, summary:<path>, artifacts:<path>, history, history:summary, history:lastLoss, or history:sort=-trainLoss:top=3.");
    }

    private static Map<String, Object> summarizeStatus(InputBundle input, boolean ci) {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", ci ? "short" : "full");
        status.put("historyFile", input.historyFile().toAbsolutePath().normalize().toString());
        status.put("reportAvailable", input.reportFile() != null);
        if (input.reportFile() == null) {
            status.put("mode", "history-only");
            Object historySummary = ByteLatentHistoryReports.select(input.historyFile(), "history:summary");
            if (ci && historySummary instanceof Map<?, ?> map) {
                status.put("historyStatus", flattenHistorySummary(map));
            } else {
                status.put("historyStatus", historySummary);
            }
            return status;
        }
        status.put("mode", "report-backed");
        Map<String, Object> summary = asMap(input.report().get("summary"));
        Map<String, Object> artifacts = asMap(input.report().get("artifacts"));
        Map<String, Object> metadata = asMap(summary.get("metadata"));
        status.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        status.put("historyCount", input.report().getOrDefault("historyCount", 0));
        status.put("globalStep", metadata.getOrDefault("globalStep", ""));
        status.put("resumeLoaded", metadata.getOrDefault("resumeLoaded", ""));
        status.put("latestTrainLoss", summary.getOrDefault("latestTrainLoss", ""));
        status.put("bestValidationLoss", summary.getOrDefault("bestValidationLoss", ""));
        status.put("checkpointDir", artifacts.getOrDefault("checkpointDir", ""));
        return status;
    }

    private static Map<String, Object> flattenHistorySummary(Map<?, ?> map) {
        Map<String, Object> flattened = new LinkedHashMap<>();
        map.forEach((key, value) -> flattened.put(String.valueOf(key), value));
        return flattened;
    }

    private static Map<String, Object> loadJsonMap(Path file) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> values = JSON.readValue(file.toFile(), LinkedHashMap.class);
            return Map.copyOf(values);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load JSON file " + file + ".", exception);
        }
    }

    private static List<ByteLatentHistoryRow> historyRows(Map<String, Object> report) {
        Object value = report.get("history");
        if (!(value instanceof List<?> list)) {
            return List.of();
        }
        List<ByteLatentHistoryRow> rows = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> row)) {
                continue;
            }
            rows.add(new ByteLatentHistoryRow(
                    asInt(row.get("epoch")),
                    asInt(row.get("globalStep")),
                    asInt(row.get("batchCount")),
                    asDouble(row.get("trainLoss"))));
        }
        return List.copyOf(rows);
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

    private static int asInt(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(String.valueOf(value));
    }

    private static double asDouble(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return Double.parseDouble(String.valueOf(value));
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
                && map.containsKey("historyFile")
                && map.containsKey("reportAvailable");
    }

    private static String renderStatusText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        boolean reportAvailable = Boolean.parseBoolean(String.valueOf(map.getOrDefault("reportAvailable", false)));
        String historyFile = String.valueOf(map.getOrDefault("historyFile", ""));
        if ("short".equals(summary)) {
            StringBuilder out = new StringBuilder();
            out.append("status")
                    .append(" report=").append(reportAvailable)
                    .append(" mode=").append(String.valueOf(map.getOrDefault("mode", "")));
            if (map.containsKey("historyCount")) {
                out.append(" historyCount=").append(String.valueOf(map.getOrDefault("historyCount", 0)))
                        .append(" globalStep=").append(String.valueOf(map.getOrDefault("globalStep", "")))
                        .append(" latestTrainLoss=").append(String.valueOf(map.getOrDefault("latestTrainLoss", "")));
            }
            out.append(" historyFile=").append(historyFile);
            return out.append('\n').toString();
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
        out.append(String.join(",", headers.stream().map(trainer_byte_latent_history_inspector::escapeCsv).toList()))
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
        String section = args.length >= 2 ? args[1] : "summary";
        String format = args.length >= 3 ? args[2] : null;
        Path outputPath = args.length >= 4 ? Path.of(args[3]) : null;
        return new CliOptions(section, format, outputPath);
    }

    private record CliOptions(String section, String format, Path outputPath) {
    }

    private record InputBundle(Path inputPath, Path historyFile, Path reportFile, Map<String, Object> report) {
    }
}
