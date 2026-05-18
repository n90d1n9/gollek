package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class TrainingHistoryCsv {
    private static final List<String> BASE_COLUMNS = List.of(
            "epoch",
            "trainLoss",
            "validationLoss",
            "learningRate",
            "optimizerStepCount",
            "schedulerStepCount",
            "bestModelMonitor",
            "bestModelMonitorMode",
            "bestModelMonitorValue");

    private TrainingHistoryCsv() {
    }

    static String write(List<Map<String, Object>> rows) {
        List<String> columns = columns(rows);
        StringBuilder csv = new StringBuilder();
        appendCsvRow(csv, columns);
        for (Map<String, Object> row : rows) {
            List<String> values = new ArrayList<>(columns.size());
            for (String column : columns) {
                values.add(csvValue(row.get(column)));
            }
            appendCsvRow(csv, values);
        }
        return csv.toString();
    }

    static List<Map<String, Object>> read(Path csvFile) throws IOException {
        String csv = Files.readString(csvFile, StandardCharsets.UTF_8);
        List<List<String>> records = parseCsv(csv);
        if (records.isEmpty()) {
            return List.of();
        }
        List<String> columns = validateColumns(records.get(0));
        List<Map<String, Object>> rows = new ArrayList<>();
        Set<Integer> seenEpochs = new LinkedHashSet<>();
        for (int rowIndex = 1; rowIndex < records.size(); rowIndex++) {
            List<String> record = records.get(rowIndex);
            if (record.isEmpty() || record.stream().allMatch(String::isBlank)) {
                continue;
            }
            if (record.size() > columns.size()) {
                throw new IOException("Invalid training history CSV row " + (rowIndex + 1)
                        + ": expected at most " + columns.size()
                        + " cells but found " + record.size());
            }
            Map<String, Object> row = new LinkedHashMap<>();
            for (int columnIndex = 0; columnIndex < columns.size() && columnIndex < record.size(); columnIndex++) {
                String column = columns.get(columnIndex);
                String rawValue = record.get(columnIndex);
                if (column == null || column.isBlank() || rawValue == null || rawValue.isBlank()) {
                    continue;
                }
                row.put(column, parseValue(column, rawValue));
            }
            if (!row.isEmpty()) {
                int epoch = validateEpoch(row, rowIndex + 1, seenEpochs);
                row.put("epoch", epoch);
                restoreNestedMetrics(row, "trainMetric.", "trainMetrics");
                restoreNestedMetrics(row, "validationMetric.", "validationMetrics");
                restoreNestedMetrics(row, "trainMetricDetails.", "trainMetricDetails");
                restoreNestedMetrics(row, "validationMetricDetails.", "validationMetricDetails");
                rows.add(row);
            }
        }
        return rows;
    }

    static Map<String, Object> copyRow(Map<String, Object> row) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Map<?, ?> rawMap) {
                Map<String, Object> nested = new LinkedHashMap<>();
                for (Map.Entry<?, ?> nestedEntry : rawMap.entrySet()) {
                    nested.put(String.valueOf(nestedEntry.getKey()), nestedEntry.getValue());
                }
                copy.put(entry.getKey(), Collections.unmodifiableMap(nested));
            } else {
                copy.put(entry.getKey(), value);
            }
        }
        return copy;
    }

    private static List<String> columns(List<Map<String, Object>> rows) {
        Set<String> columns = new LinkedHashSet<>(BASE_COLUMNS);
        for (Map<String, Object> row : rows) {
            for (Map.Entry<String, Object> entry : row.entrySet()) {
                if (entry.getValue() != null) {
                    columns.add(entry.getKey());
                }
            }
        }
        return List.copyOf(columns);
    }

    private static String csvValue(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof Map<?, ?> || value instanceof Iterable<?> || value.getClass().isArray()) {
            return TrainerJson.toJson(value);
        }
        if (value instanceof Number number) {
            if (isNonFiniteNumber(number)) {
                return "null";
            }
            return number.toString();
        }
        if (value instanceof Boolean bool) {
            return bool.toString();
        }
        return String.valueOf(value);
    }

    private static void appendCsvRow(StringBuilder csv, List<String> values) {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                csv.append(',');
            }
            appendCsvCell(csv, values.get(i));
        }
        csv.append('\n');
    }

    private static void appendCsvCell(StringBuilder csv, String value) {
        boolean quote = value.indexOf(',') >= 0
                || value.indexOf('"') >= 0
                || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0;
        if (!quote) {
            csv.append(value);
            return;
        }
        csv.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (ch == '"') {
                csv.append('"');
            }
            csv.append(ch);
        }
        csv.append('"');
    }

    private static List<String> validateColumns(List<String> rawColumns) throws IOException {
        if (rawColumns == null || rawColumns.isEmpty()) {
            throw new IOException("Invalid training history CSV header: missing columns");
        }
        List<String> columns = new ArrayList<>(rawColumns.size());
        Set<String> seen = new LinkedHashSet<>();
        for (int index = 0; index < rawColumns.size(); index++) {
            String column = rawColumns.get(index) == null ? "" : rawColumns.get(index).trim();
            if (column.isEmpty()) {
                throw new IOException("Invalid training history CSV header: blank column at index " + index);
            }
            if (!seen.add(column)) {
                throw new IOException("Invalid training history CSV header: duplicate column '" + column
                        + "' at index " + index);
            }
            columns.add(column);
        }
        if (!seen.contains("epoch")) {
            throw new IOException("Invalid training history CSV header: missing required 'epoch' column");
        }
        return List.copyOf(columns);
    }

    private static int validateEpoch(
            Map<String, Object> row,
            int rowNumber,
            Set<Integer> seenEpochs) throws IOException {
        Object rawEpoch = row.get("epoch");
        if (!(rawEpoch instanceof Number number)) {
            throw new IOException("Invalid training history CSV row " + rowNumber
                    + ": epoch is missing or not numeric");
        }
        double epochValue = number.doubleValue();
        if (!Double.isFinite(epochValue)
                || epochValue < 0.0
                || epochValue > Integer.MAX_VALUE
                || Math.rint(epochValue) != epochValue) {
            throw new IOException("Invalid training history CSV row " + rowNumber
                    + ": epoch must be a non-negative integer");
        }
        int epoch = (int) epochValue;
        if (!seenEpochs.add(epoch)) {
            throw new IOException("Invalid training history CSV row " + rowNumber
                    + ": duplicate epoch " + epoch);
        }
        return epoch;
    }

    private static List<List<String>> parseCsv(String csv) throws IOException {
        List<List<String>> records = new ArrayList<>();
        List<String> record = new ArrayList<>();
        StringBuilder cell = new StringBuilder();
        boolean quoted = false;
        for (int index = 0; index < csv.length(); index++) {
            char ch = csv.charAt(index);
            if (quoted) {
                if (ch == '"') {
                    if (index + 1 < csv.length() && csv.charAt(index + 1) == '"') {
                        cell.append('"');
                        index++;
                    } else {
                        quoted = false;
                    }
                } else {
                    cell.append(ch);
                }
                continue;
            }
            if (ch == '"') {
                quoted = true;
            } else if (ch == ',') {
                record.add(cell.toString());
                cell.setLength(0);
            } else if (ch == '\n') {
                record.add(cell.toString());
                cell.setLength(0);
                records.add(record);
                record = new ArrayList<>();
            } else if (ch != '\r') {
                cell.append(ch);
            }
        }
        if (quoted) {
            throw new IOException("Invalid training history CSV: unterminated quoted cell");
        }
        if (cell.length() > 0 || !record.isEmpty()) {
            record.add(cell.toString());
            records.add(record);
        }
        return records;
    }

    private static Object parseValue(String column, String rawValue) throws IOException {
        String value = rawValue.trim();
        if (isJsonObjectColumn(column)) {
            Object parsed = parseRequiredJson(column, value);
            if (!(parsed instanceof Map<?, ?>)) {
                throw new IOException("Invalid JSON in training history column '" + column
                        + "': expected JSON object");
            }
            return parsed;
        }
        if (isStructuredDetailColumn(column) && looksLikeJsonValue(value)) {
            return parseRequiredJson(column, value);
        }
        return parseScalar(rawValue);
    }

    private static Object parseScalar(String rawValue) {
        String value = rawValue.trim();
        if (looksLikeJsonValue(value)) {
            try {
                return TrainerJsonParser.parse(value);
            } catch (IllegalArgumentException ignored) {
                // Keep backward compatibility with any hand-edited CSV cells
                // that start like JSON but are intended as plain strings.
            }
        }
        if ("null".equalsIgnoreCase(value)) {
            return null;
        }
        if ("true".equalsIgnoreCase(value)) {
            return Boolean.TRUE;
        }
        if ("false".equalsIgnoreCase(value)) {
            return Boolean.FALSE;
        }
        try {
            if (!value.contains(".") && !value.contains("e") && !value.contains("E")) {
                long parsed = Long.parseLong(value);
                if (parsed >= Integer.MIN_VALUE && parsed <= Integer.MAX_VALUE) {
                    return (int) parsed;
                }
                return parsed;
            }
            return Double.parseDouble(value);
        } catch (NumberFormatException integerError) {
            try {
                return Double.parseDouble(value);
            } catch (NumberFormatException floatingPointError) {
                return rawValue;
            }
        }
    }

    private static Object parseRequiredJson(String column, String value) throws IOException {
        if (!looksLikeJsonValue(value)) {
            throw new IOException("Invalid JSON in training history column '" + column
                    + "': expected JSON value");
        }
        try {
            return TrainerJsonParser.parse(value);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid JSON in training history column '" + column
                    + "': " + error.getMessage(), error);
        }
    }

    private static void restoreNestedMetrics(
            Map<String, Object> row,
            String metricPrefix,
            String metricMapKey) {
        Map<String, Object> metrics = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            if (entry.getKey().startsWith(metricPrefix)) {
                metrics.put(entry.getKey().substring(metricPrefix.length()), entry.getValue());
            }
        }
        if (!metrics.isEmpty()) {
            row.put(metricMapKey, Collections.unmodifiableMap(metrics));
        }
    }

    private static boolean isNonFiniteNumber(Number number) {
        if (number instanceof Double doubleValue) {
            return !Double.isFinite(doubleValue.doubleValue());
        }
        if (number instanceof Float floatValue) {
            return !Float.isFinite(floatValue.floatValue());
        }
        return false;
    }

    private static boolean isJsonObjectColumn(String column) {
        return "trainMetrics".equals(column)
                || "validationMetrics".equals(column)
                || "trainMetricDetails".equals(column)
                || "validationMetricDetails".equals(column);
    }

    private static boolean isStructuredDetailColumn(String column) {
        return column != null
                && (column.startsWith("trainMetricDetails.")
                        || column.startsWith("validationMetricDetails."));
    }

    private static boolean looksLikeJsonValue(String value) {
        return value.startsWith("{")
                || value.startsWith("[")
                || value.startsWith("\"");
    }

}
