package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

final class TrainerCsv {
    private TrainerCsv() {
    }

    static String toCsv(List<String> columns, List<? extends Map<String, ?>> rows) {
        StringBuilder csv = new StringBuilder();
        appendRow(csv, columns);
        for (Map<String, ?> row : rows) {
            appendRow(csv, columns.stream()
                    .map(column -> csvValue(row.get(column)))
                    .toList());
        }
        return csv.toString();
    }

    static String csvValue(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? number.toString() : "";
        }
        if (value instanceof Map<?, ?> || value instanceof Iterable<?>) {
            return TrainerJson.toJson(value);
        }
        return String.valueOf(value);
    }

    private static void appendRow(StringBuilder csv, List<String> values) {
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                csv.append(',');
            }
            appendCell(csv, values.get(i));
        }
        csv.append('\n');
    }

    private static void appendCell(StringBuilder csv, String value) {
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
}
