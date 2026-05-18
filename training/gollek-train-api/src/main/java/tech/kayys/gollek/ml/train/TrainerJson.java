package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

final class TrainerJson {
    private TrainerJson() {
    }

    static String toJson(Object value) {
        StringBuilder json = new StringBuilder();
        appendJson(json, value);
        return json.toString();
    }

    private static void appendJson(StringBuilder json, Object value) {
        if (value == null) {
            json.append("null");
        } else if (value instanceof String text) {
            appendJsonString(json, text);
        } else if (value instanceof Number number) {
            appendJsonNumber(json, number);
        } else if (value instanceof Boolean bool) {
            json.append(bool.booleanValue());
        } else if (value instanceof Map<?, ?> map) {
            appendJsonMap(json, map);
        } else if (value instanceof Iterable<?> iterable) {
            appendJsonIterable(json, iterable);
        } else if (value.getClass().isArray()) {
            appendJsonArray(json, value);
        } else {
            appendJsonString(json, String.valueOf(value));
        }
    }

    private static void appendJsonMap(StringBuilder json, Map<?, ?> map) {
        json.append('{');
        List<Map.Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Comparator.comparing(entry -> String.valueOf(entry.getKey())));
        for (int i = 0; i < entries.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            Map.Entry<?, ?> entry = entries.get(i);
            appendJsonString(json, String.valueOf(entry.getKey()));
            json.append(':');
            appendJson(json, entry.getValue());
        }
        json.append('}');
    }

    private static void appendJsonIterable(StringBuilder json, Iterable<?> values) {
        json.append('[');
        int index = 0;
        for (Object value : values) {
            if (index++ > 0) {
                json.append(',');
            }
            appendJson(json, value);
        }
        json.append(']');
    }

    private static void appendJsonArray(StringBuilder json, Object array) {
        json.append('[');
        int length = java.lang.reflect.Array.getLength(array);
        for (int i = 0; i < length; i++) {
            if (i > 0) {
                json.append(',');
            }
            appendJson(json, java.lang.reflect.Array.get(array, i));
        }
        json.append(']');
    }

    private static void appendJsonNumber(StringBuilder json, Number number) {
        if (number instanceof Double doubleValue && !Double.isFinite(doubleValue.doubleValue())) {
            json.append("null");
            return;
        }
        if (number instanceof Float floatValue && !Float.isFinite(floatValue.floatValue())) {
            json.append("null");
            return;
        }
        json.append(number);
    }

    private static void appendJsonString(StringBuilder json, String value) {
        json.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"' -> json.append("\\\"");
                case '\\' -> json.append("\\\\");
                case '\b' -> json.append("\\b");
                case '\f' -> json.append("\\f");
                case '\n' -> json.append("\\n");
                case '\r' -> json.append("\\r");
                case '\t' -> json.append("\\t");
                default -> {
                    if (ch < 0x20) {
                        json.append(String.format("\\u%04x", (int) ch));
                    } else {
                        json.append(ch);
                    }
                }
            }
        }
        json.append('"');
    }
}
