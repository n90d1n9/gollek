package tech.kayys.gollek.train.diffusion.opd;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Owns table/CSV rendering and row projection for the inspector CLI.
 *
 * <p>This helper is responsible for format-local layout and normalization after
 * section routing has already been resolved by the support or manifest helpers.
 */
final class DiffusionOpdReportInspectorRendering {

    private DiffusionOpdReportInspectorRendering() {
    }

    static String renderByFormat(String name, Object value, String format, String columns) {
        return switch (format) {
            case "json" -> DiffusionOpdReportInspectorSupport.renderJson(value);
            case "table" -> renderTable(value, columns);
            case "csv" -> renderCsv(value, columns);
            case "text" -> name + "=" + value + System.lineSeparator();
            default -> throw new IllegalArgumentException(
                    "Unknown format '" + format + "'. Use text, json, table, or csv.");
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
            values.add(DiffusionOpdReportInspectorSupport.escapeCsv(stringifyCell(row.get(header))));
        }
        return String.join(",", values);
    }

    private static RenderGrid prepareRenderGrid(Object value, String columns) {
        List<Map<String, Object>> rows = prepareProjectedRows(value, columns);
        return new RenderGrid(rows, headers(rows));
    }

    private static List<Map<String, Object>> prepareProjectedRows(Object value, String columns) {
        return projectRows(toRows(value), columns);
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

    private static Map<String, Object> mergeRowMaps(Map<String, Object> base, Map<String, Object> additions) {
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

    private static Object resolveProjectedValue(Map<String, Object> row, String column) {
        String resolvedColumn = resolveColumnAlias(column);
        if (isDirectProjectedColumn(resolvedColumn)) {
            return row.get(resolvedColumn);
        }
        return descendProjectedPath(row, resolvedColumn);
    }

    private static String resolveColumnAlias(String column) {
        return DiffusionOpdReportInspectorSupport.COLUMN_ALIASES.getOrDefault(column, column);
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

    static Map<String, Object> normalizeMap(Map<?, ?> map) {
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
        if (value instanceof Number || value instanceof Boolean || value instanceof CharSequence) {
            return String.valueOf(value);
        }
        return DiffusionOpdReportInspectorSupport.stringifyStructuredCell(value);
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
}
