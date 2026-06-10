package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

final class DiscreteTokenDatasetSchemaMetadata {
    private DiscreteTokenDatasetSchemaMetadata() {
    }

    static Map<String, Object> immutableStringMap(Map<?, ?> source, String field) {
        Objects.requireNonNull(source, field + " must not be null");
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            Object key = entry.getKey();
            if (!(key instanceof String text) || text.isBlank()) {
                throw new IllegalArgumentException(field + " must contain only non-blank string keys");
            }
            copy.put(text, entry.getValue());
        }
        return Collections.unmodifiableMap(copy);
    }

    static List<Map<String, Object>> requiredMapList(Map<?, ?> metadata, String field) {
        Object value = metadata.get(field);
        if (!(value instanceof List<?> list)) {
            throw new IllegalArgumentException(field + " must be an array");
        }
        List<Map<String, Object>> copy = new ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException(field + " must contain objects");
            }
            copy.add(immutableStringMap(map, field + " item"));
        }
        return List.copyOf(copy);
    }

    static List<String> requiredStringList(Map<?, ?> metadata, String field) {
        Object value = metadata.get(field);
        if (!(value instanceof List<?> list)) {
            throw new IllegalArgumentException(field + " must be an array");
        }
        List<String> copy = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof String text && !text.isBlank()) {
                copy.add(text);
            } else {
                throw new IllegalArgumentException(field + " must contain non-blank strings");
            }
        }
        return List.copyOf(copy);
    }

    static List<String> optionalStringList(Map<?, ?> metadata, String field) {
        if (!metadata.containsKey(field) || metadata.get(field) == null) {
            return List.of();
        }
        return requiredStringList(metadata, field);
    }

    static String requiredString(Map<?, ?> metadata, String field) {
        Object value = metadata.get(field);
        if (value instanceof String text && !text.isBlank()) {
            return text;
        }
        throw new IllegalArgumentException(field + " must be a non-blank string");
    }

    static Optional<String> optionalString(Map<?, ?> metadata, String field) {
        if (!metadata.containsKey(field) || metadata.get(field) == null) {
            return Optional.empty();
        }
        return Optional.of(requiredString(metadata, field));
    }

    static boolean requiredBoolean(Map<?, ?> metadata, String field) {
        Object value = metadata.get(field);
        if (value instanceof Boolean flag) {
            return flag;
        }
        throw new IllegalArgumentException(field + " must be a boolean");
    }

    static int requiredInt(Map<?, ?> metadata, String field) {
        Object value = metadata.get(field);
        if (value instanceof Number number) {
            int intValue = number.intValue();
            if (number.longValue() == intValue) {
                return intValue;
            }
        }
        throw new IllegalArgumentException(field + " must be an integer");
    }
}
