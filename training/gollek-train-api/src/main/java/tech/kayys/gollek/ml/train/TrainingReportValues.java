package tech.kayys.gollek.ml.train;

import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Shared scalar coercion helpers for typed training-report views.
 */
final class TrainingReportValues {
    private TrainingReportValues() {
    }

    static OptionalDouble optionalDouble(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? OptionalDouble.of(doubleValue) : OptionalDouble.empty();
        }
        if (value instanceof String text) {
            try {
                double doubleValue = Double.parseDouble(text);
                return Double.isFinite(doubleValue) ? OptionalDouble.of(doubleValue) : OptionalDouble.empty();
            } catch (NumberFormatException ignored) {
                return OptionalDouble.empty();
            }
        }
        return OptionalDouble.empty();
    }

    static OptionalInt optionalInt(Object value) {
        if (value instanceof Number number) {
            return OptionalInt.of(number.intValue());
        }
        if (value instanceof String text) {
            try {
                return OptionalInt.of(Integer.parseInt(text));
            } catch (NumberFormatException ignored) {
                return OptionalInt.empty();
            }
        }
        return OptionalInt.empty();
    }

    static OptionalLong optionalLong(Object value) {
        if (value instanceof Number number) {
            return OptionalLong.of(number.longValue());
        }
        if (value instanceof String text) {
            try {
                return OptionalLong.of(Long.parseLong(text));
            } catch (NumberFormatException ignored) {
                return OptionalLong.empty();
            }
        }
        return OptionalLong.empty();
    }

    static int intValue(Object value, int fallback) {
        OptionalInt parsed = optionalInt(value);
        return parsed.isPresent() ? parsed.getAsInt() : fallback;
    }

    static long longValue(Object value, long fallback) {
        OptionalLong parsed = optionalLong(value);
        return parsed.isPresent() ? parsed.getAsLong() : fallback;
    }

    static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool.booleanValue();
        }
        if (value instanceof String text) {
            return Boolean.parseBoolean(text);
        }
        return false;
    }

    static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    static String normalizedString(String value, String fallback) {
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return value.trim();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> mapValue(Map<String, ?> source, String key) {
        Objects.requireNonNull(source, "source must not be null");
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }
}
