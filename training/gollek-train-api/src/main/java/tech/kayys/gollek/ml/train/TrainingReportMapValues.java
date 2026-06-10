package tech.kayys.gollek.ml.train;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Typed accessors for raw JSON-like training report maps.
 */
final class TrainingReportMapValues {
    private TrainingReportMapValues() {
    }

    static Optional<String> stringValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof String text && !text.isBlank()) {
            return Optional.of(text);
        }
        return Optional.empty();
    }

    static String stringValue(Map<String, ?> map, String key, String fallback) {
        return stringValue(map, key).orElse(fallback);
    }

    static Optional<Map<String, Object>> objectValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Map<?, ?> object) {
            return Optional.of(immutableMap(object));
        }
        return Optional.empty();
    }

    static Optional<Path> pathValue(Map<String, ?> map, String key) {
        return stringValue(map, key).flatMap(TrainingReportMapValues::pathValue);
    }

    static Optional<Path> pathValue(String value) {
        if (value == null || value.isBlank()) {
            return Optional.empty();
        }
        try {
            return Optional.of(Path.of(value).toAbsolutePath().normalize());
        } catch (InvalidPathException ignored) {
            return Optional.empty();
        }
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
            if (value instanceof String text) {
                strings.add(text);
            } else {
                failures.add(owner + "." + key + "[" + index + "] must be a string");
            }
            index++;
        }
        return Optional.of(List.copyOf(strings));
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

    static boolean booleanValue(Map<String, ?> map, String key, boolean fallback) {
        return booleanValue(map, key).orElse(fallback);
    }

    static Optional<Long> longValue(Map<String, ?> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        if (value instanceof String text && !text.isBlank()) {
            try {
                return Optional.of(Long.parseLong(text.trim()));
            } catch (NumberFormatException ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    static long longValue(Map<String, ?> map, String key, long fallback) {
        return longValue(map, key).orElse(fallback);
    }

    static boolean isSha256Hex(String value) {
        if (value == null || value.length() != 64) {
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

    static String requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return checksum.trim().toLowerCase();
    }

    static Map<String, Object> immutableMap(Map<?, ?> map) {
        return TrainingReportSnapshots.immutableMap(map);
    }
}
