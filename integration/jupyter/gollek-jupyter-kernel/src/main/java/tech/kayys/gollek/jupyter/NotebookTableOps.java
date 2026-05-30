package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookTables.getCell;
import static tech.kayys.gollek.jupyter.NotebookTables.parseFiniteDouble;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

final class NotebookTableOps {

    private NotebookTableOps() {
    }

    record SortRow(List<String> cells, String key, Double numericKey, int originalIndex) {}

    static List<List<String>> sampleRows(List<List<String>> rows, int requestedRows, Long seed) {
        if (rows.size() <= requestedRows) {
            return rows;
        }
        List<List<String>> shuffled = new ArrayList<>(rows);
        if (seed == null) {
            Collections.shuffle(shuffled);
        } else {
            Collections.shuffle(shuffled, new Random(seed));
        }
        return shuffled.stream().limit(requestedRows).toList();
    }

    static boolean isNumericSortColumn(List<List<String>> rows, int columnIndex) {
        boolean found = false;
        for (List<String> row : rows) {
            String value = getCell(row, columnIndex).trim();
            if (value.isEmpty()) {
                continue;
            }
            found = true;
            if (parseFiniteDouble(value) == null) {
                return false;
            }
        }
        return found;
    }

    static int compareSortRows(SortRow left, SortRow right, boolean numeric, boolean descending) {
        boolean leftBlank = left.key().isEmpty();
        boolean rightBlank = right.key().isEmpty();
        if (leftBlank && rightBlank) {
            return Integer.compare(left.originalIndex(), right.originalIndex());
        }
        if (leftBlank) {
            return 1;
        }
        if (rightBlank) {
            return -1;
        }
        int result;
        if (numeric) {
            result = Double.compare(left.numericKey(), right.numericKey());
        } else {
            result = String.CASE_INSENSITIVE_ORDER.compare(left.key(), right.key());
            if (result == 0) {
                result = left.key().compareTo(right.key());
            }
        }
        if (descending) {
            result = -result;
        }
        return result == 0 ? Integer.compare(left.originalIndex(), right.originalIndex()) : result;
    }

    static String normalizeFilterOperator(String operator) {
        String normalized = operator.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "=", "==", "eq" -> "==";
            case "!=", "<>", "ne" -> "!=";
            case ">", "gt" -> ">";
            case ">=", "gte", "ge" -> ">=";
            case "<", "lt" -> "<";
            case "<=", "lte", "le" -> "<=";
            case "contains" -> "contains";
            case "!contains", "notcontains", "not_contains" -> "!contains";
            case "starts", "startswith", "starts_with" -> "starts";
            case "ends", "endswith", "ends_with" -> "ends";
            case "blank", "isblank", "is_blank" -> "blank";
            case "notblank", "nonblank", "not_blank" -> "notblank";
            default -> throw new IllegalArgumentException("Unknown %filter operator: " + operator);
        };
    }

    static boolean filterRequiresValue(String operator) {
        return !operator.equals("blank") && !operator.equals("notblank");
    }

    static boolean matchesFilter(String rawCell, String operator, String rawValue) {
        String cell = rawCell == null ? "" : rawCell.trim();
        String value = rawValue == null ? "" : rawValue.trim();
        return switch (operator) {
            case "==" -> cell.equals(value);
            case "!=" -> !cell.equals(value);
            case "contains" -> cell.contains(value);
            case "!contains" -> !cell.contains(value);
            case "starts" -> cell.startsWith(value);
            case "ends" -> cell.endsWith(value);
            case "blank" -> cell.isEmpty();
            case "notblank" -> !cell.isEmpty();
            case ">", ">=", "<", "<=" -> matchesNumericFilter(cell, operator, value);
            default -> false;
        };
    }

    static String filterPredicateLabel(String column, String operator, String value) {
        return value == null
                ? column + " " + operator
                : column + " " + operator + " " + value;
    }

    private static boolean matchesNumericFilter(String cell, String operator, String value) {
        Double left = parseFiniteDouble(cell);
        Double right = parseFiniteDouble(value);
        if (left == null || right == null) {
            return false;
        }
        return switch (operator) {
            case ">" -> left > right;
            case ">=" -> left >= right;
            case "<" -> left < right;
            case "<=" -> left <= right;
            default -> false;
        };
    }
}
