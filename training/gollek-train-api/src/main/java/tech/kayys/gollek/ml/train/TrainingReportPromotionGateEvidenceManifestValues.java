package tech.kayys.gollek.ml.train;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Typed accessors for raw evidence manifest maps.
 */
final class TrainingReportPromotionGateEvidenceManifestValues {
    private TrainingReportPromotionGateEvidenceManifestValues() {
    }

    static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Map<?, ?> object) {
            return Optional.of(immutableMap(object));
        }
        return Optional.empty();
    }

    static Optional<Long> longValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Optional.of(Long.parseLong(text));
            } catch (NumberFormatException ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    static Optional<Path> pathValue(Map<String, ?> map, String key) {
        Optional<String> value = stringValue(map, key);
        if (value.isEmpty()) {
            return Optional.empty();
        }
        try {
            return Optional.of(Path.of(value.orElseThrow()).toAbsolutePath().normalize());
        } catch (InvalidPathException error) {
            return Optional.empty();
        }
    }

    static Optional<Boolean> booleanValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Boolean flag) {
            return Optional.of(flag);
        }
        if (value instanceof String text && !text.isBlank()) {
            String normalized = text.trim().toLowerCase();
            if ("true".equals(normalized)) {
                return Optional.of(Boolean.TRUE);
            }
            if ("false".equals(normalized)) {
                return Optional.of(Boolean.FALSE);
            }
        }
        return Optional.empty();
    }

    static Optional<List<?>> iterableValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof List<?> list) {
            return Optional.of(List.copyOf(list));
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> values = new ArrayList<>();
            for (Object item : iterable) {
                values.add(item);
            }
            return Optional.of(List.copyOf(values));
        }
        return Optional.empty();
    }

    static Optional<List<String>> stringListValue(
            Map<String, ?> map,
            String key,
            String owner,
            List<String> failures) {
        Optional<List<?>> values = iterableValue(map, key);
        if (values.isEmpty()) {
            return Optional.empty();
        }
        List<String> strings = new ArrayList<>();
        int index = 0;
        for (Object value : values.orElseThrow()) {
            if (value instanceof String text && !text.isBlank()) {
                strings.add(text);
            } else {
                failures.add(owner + "." + key + "[" + index + "] must be a non-blank string");
            }
            index++;
        }
        return Optional.of(List.copyOf(strings));
    }

    static void rejectDuplicateStrings(
            List<String> values,
            String owner,
            List<String> failures) {
        Set<String> seen = new LinkedHashSet<>();
        for (String value : values) {
            if (!seen.add(value)) {
                failures.add(owner + " contains duplicate entry " + value);
            }
        }
    }

    static List<String> difference(List<String> left, List<String> right) {
        Set<String> rightSet = new LinkedHashSet<>(right);
        List<String> values = new ArrayList<>();
        for (String value : left) {
            if (!rightSet.contains(value)) {
                values.add(value);
            }
        }
        return List.copyOf(values);
    }

    static boolean isSha256Hex(String value) {
        if (value.length() != 64) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            boolean hex = (ch >= '0' && ch <= '9')
                    || (ch >= 'a' && ch <= 'f')
                    || (ch >= 'A' && ch <= 'F');
            if (!hex) {
                return false;
            }
        }
        return true;
    }

    static Map<String, Object> immutableMap(Map<?, ?> map) {
        return TrainingReportSnapshots.immutableMap(map);
    }
}
