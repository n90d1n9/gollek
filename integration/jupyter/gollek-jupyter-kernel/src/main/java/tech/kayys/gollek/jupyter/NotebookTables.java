package tech.kayys.gollek.jupyter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

final class NotebookTables {

    private NotebookTables() {
    }

    static String formatDelimitedPreviewRow(List<String> row, int width) {
        List<String> cells = new ArrayList<>();
        for (int i = 0; i < width; i++) {
            cells.add(getCell(row, i));
        }
        return String.join(" | ", cells);
    }

    static String getCell(List<String> row, int column) {
        return column < row.size() ? row.get(column) : "";
    }

    static Double parseFiniteDouble(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return null;
        }
        try {
            double value = Double.parseDouble(raw.trim());
            return Double.isFinite(value) ? value : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    static String formatNullableStat(Double value) {
        if (value == null) {
            return "";
        }
        if (Math.rint(value) == value) {
            return Long.toString(value.longValue());
        }
        return String.format(Locale.ROOT, "%.4f", value);
    }

    static String formatPercent(int count, int total) {
        if (total <= 0) {
            return "0.00%";
        }
        return String.format(Locale.ROOT, "%.2f%%", count * 100.0 / total);
    }

    static List<String> parseCsvLine(String line) {
        List<String> cells = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                cells.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }
        cells.add(current.toString());
        return cells;
    }

    static List<String> parseDelimitedLine(String line, String delimiter) {
        return Arrays.asList(line.split(Pattern.quote(delimiter), -1));
    }
}
