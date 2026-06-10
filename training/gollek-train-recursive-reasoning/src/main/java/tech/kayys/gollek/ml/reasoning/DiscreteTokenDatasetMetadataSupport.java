package tech.kayys.gollek.ml.reasoning;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Shared metadata parsing helpers for recursive-reasoning checkpoint payloads.
 */
final class DiscreteTokenDatasetMetadataSupport {
    private DiscreteTokenDatasetMetadataSupport() {}

    static Object required(Map<?, ?> metadata, String key) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException("metadata field '" + key + "' is required");
        }
        return metadata.get(key);
    }

    static Object required(Map<?, ?> metadata, String key, String owner) {
        if (!metadata.containsKey(key) || metadata.get(key) == null) {
            throw new IllegalArgumentException(owner + " field '" + key + "' is required");
        }
        return metadata.get(key);
    }

    static Map<?, ?> requiredMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return map;
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    static Map<String, Object> requiredMetadataMap(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Map<?, ?> map) {
            return immutableMetadataMap(map, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    static Map<String, Object> optionalMetadataMap(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return Map.of();
        }
        if (value instanceof Map<?, ?> map) {
            return immutableMetadataMap(map, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    static List<Map<String, Object>> requiredMetadataMapList(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof List<?> list) {
            return immutableMetadataMapList(list, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    static List<Map<String, Object>> immutableMetadataMapList(List<?> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof Map<?, ?> map) {
                        return immutableMetadataMap(map, name + " entry");
                    }
                    throw new IllegalArgumentException(name + " entries must be maps");
                })
                .toList();
    }

    static Map<String, Object> immutableMetadataMap(Map<?, ?> values, String name) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            String key = mapKey(entry.getKey(), name);
            Object value = Objects.requireNonNull(entry.getValue(), name + " field '" + key + "' must not be null");
            copy.put(key, value);
        }
        return Collections.unmodifiableMap(copy);
    }

    static Map<String, Object> optionalJsonMetadataMap(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return Map.of();
        }
        if (value instanceof Map<?, ?> map) {
            return immutableJsonMetadataMap(map, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a map");
    }

    static Map<String, Object> immutableJsonMetadataMap(Map<?, ?> values, String name) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            String key = mapKey(entry.getKey(), name);
            Object value = Objects.requireNonNull(entry.getValue(), name + " field '" + key + "' must not be null");
            copy.put(key, immutableJsonMetadataValue(value, name + "." + key));
        }
        return Collections.unmodifiableMap(copy);
    }

    private static Object immutableJsonMetadataValue(Object value, String name) {
        if (value instanceof CharSequence text) {
            return text.toString();
        }
        if (value instanceof Number || value instanceof Boolean) {
            return value;
        }
        if (value instanceof Map<?, ?> map) {
            return immutableJsonMetadataMap(map, name);
        }
        if (value instanceof List<?> list) {
            return list.stream()
                    .map(entry -> immutableJsonMetadataValue(
                            Objects.requireNonNull(entry, name + " entry must not be null"),
                            name + " entry"))
                    .toList();
        }
        throw new IllegalArgumentException(name + " must be a JSON-compatible metadata value");
    }

    static String mapKey(Object key, String mapName) {
        if (key instanceof CharSequence text) {
            return requireText(text.toString(), mapName + " key");
        }
        throw new IllegalArgumentException(mapName + " keys must be strings");
    }

    static String requiredString(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof CharSequence text) {
            return requireText(text.toString(), key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    static String optionalString(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return null;
        }
        if (value instanceof CharSequence text) {
            return requireText(text.toString(), key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    static String optionalString(Map<?, ?> metadata, String key, String defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof CharSequence text) {
            return requireText(text.toString(), key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a string");
    }

    static boolean requiredBoolean(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof CharSequence text) {
            String normalized = text.toString().trim().toLowerCase();
            if ("true".equals(normalized)) {
                return true;
            }
            if ("false".equals(normalized)) {
                return false;
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a boolean");
    }

    static boolean optionalBoolean(Map<?, ?> metadata, String key, boolean defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof CharSequence text) {
            String normalized = text.toString().trim().toLowerCase();
            if ("true".equals(normalized)) {
                return true;
            }
            if ("false".equals(normalized)) {
                return false;
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a boolean");
    }

    static int requiredInt(Map<?, ?> metadata, String key) {
        long value = requiredLong(metadata, key);
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("metadata field '" + key + "' must fit in an int");
        }
        return (int) value;
    }

    static int optionalInt(Map<?, ?> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        long parsed = longValue(value, key);
        if (parsed < Integer.MIN_VALUE || parsed > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("metadata field '" + key + "' must fit in an int");
        }
        return (int) parsed;
    }

    static long requiredLong(Map<?, ?> metadata, String key) {
        return longValue(required(metadata, key), key);
    }

    static Long optionalLong(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return null;
        }
        return longValue(value, key);
    }

    private static long longValue(Object value, String key) {
        if (value instanceof Number number) {
            double numericValue = number.doubleValue();
            if (!Double.isFinite(numericValue)
                    || Math.rint(numericValue) != numericValue
                    || numericValue < Long.MIN_VALUE
                    || numericValue > Long.MAX_VALUE) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
            }
            return number.longValue();
        }
        if (value instanceof CharSequence text) {
            try {
                return Long.parseLong(text.toString());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be an integer", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be an integer");
    }

    static double requiredDouble(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof Number number) {
            double numericValue = number.doubleValue();
            if (Double.isFinite(numericValue)) {
                return numericValue;
            }
        }
        if (value instanceof CharSequence text) {
            try {
                double numericValue = Double.parseDouble(text.toString());
                if (Double.isFinite(numericValue)) {
                    return numericValue;
                }
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("metadata field '" + key + "' must be a number", e);
            }
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a number");
    }

    static List<String> requiredStringList(Map<?, ?> metadata, String key) {
        Object value = required(metadata, key);
        if (value instanceof List<?> values) {
            return immutableTextList(values, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    static List<String> optionalStringList(Map<?, ?> metadata, String key) {
        Object value = metadata.get(key);
        if (value == null) {
            return List.of();
        }
        if (value instanceof List<?> values) {
            return immutableTextList(values, key);
        }
        throw new IllegalArgumentException("metadata field '" + key + "' must be a list");
    }

    static List<String> optionalTextList(List<?> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return immutableTextList(values, name);
    }

    static List<String> immutableTextList(List<?> values, String name) {
        if (values == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        return values.stream()
                .map(value -> {
                    if (value instanceof CharSequence text) {
                        return requireText(text.toString(), name + " entry");
                    }
                    throw new IllegalArgumentException(name + " entries must be strings");
                })
                .toList();
    }

    static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    static String optionalText(String value, String name) {
        if (value == null) {
            return null;
        }
        return requireText(value, name);
    }

    static Long optionalNonNegative(Long value, String name) {
        if (value == null) {
            return null;
        }
        if (value < 0L) {
            throw new IllegalArgumentException(name + " must be >= 0 but was " + value);
        }
        return value;
    }

    static boolean metadataValueMatches(Object expected, Object actual) {
        return metadataValueMatches(expected, actual, false);
    }

    static boolean metadataValueMatches(Object expected, Object actual, boolean requireExactMapKeys) {
        return metadataValueMatches(
                expected,
                actual,
                requireExactMapKeys,
                key -> false,
                (key, expectedValue, actualValue) -> false);
    }

    static boolean metadataValueMatches(
            Object expected,
            Object actual,
            boolean requireExactMapKeys,
            MissingMetadataFieldPolicy missingFieldPolicy,
            MetadataValueMismatchPolicy mismatchPolicy) {
        Objects.requireNonNull(missingFieldPolicy, "missingFieldPolicy must not be null");
        Objects.requireNonNull(mismatchPolicy, "mismatchPolicy must not be null");
        if (expected instanceof Number expectedNumber) {
            return actual instanceof Number actualNumber
                    && Double.compare(expectedNumber.doubleValue(), actualNumber.doubleValue()) == 0;
        }
        if (expected instanceof Boolean expectedBoolean) {
            return actual instanceof Boolean actualBoolean && expectedBoolean.equals(actualBoolean);
        }
        if (expected instanceof String expectedString) {
            return actual instanceof CharSequence actualText && expectedString.contentEquals(actualText);
        }
        if (expected instanceof List<?> expectedList) {
            if (!(actual instanceof List<?> actualList) || expectedList.size() != actualList.size()) {
                return false;
            }
            for (int i = 0; i < expectedList.size(); i++) {
                if (!metadataValueMatches(
                        expectedList.get(i),
                        actualList.get(i),
                        requireExactMapKeys,
                        missingFieldPolicy,
                        mismatchPolicy)) {
                    return false;
                }
            }
            return true;
        }
        if (expected instanceof Map<?, ?> expectedMap) {
            if (!(actual instanceof Map<?, ?> actualMap)) {
                return false;
            }
            for (Map.Entry<?, ?> expectedEntry : expectedMap.entrySet()) {
                Object actualValue = actualMap.get(expectedEntry.getKey());
                if (actualValue == null) {
                    if (missingFieldPolicy.allowsMissing(expectedEntry.getKey())) {
                        continue;
                    }
                    return false;
                }
                if (!metadataValueMatches(
                        expectedEntry.getValue(),
                        actualValue,
                        requireExactMapKeys,
                        missingFieldPolicy,
                        mismatchPolicy)) {
                    if (mismatchPolicy.allowsMismatch(expectedEntry.getKey(), expectedEntry.getValue(), actualValue)) {
                        continue;
                    }
                    return false;
                }
            }
            if (requireExactMapKeys) {
                for (Object actualKey : actualMap.keySet()) {
                    if (!expectedMap.containsKey(actualKey)) {
                        return false;
                    }
                }
            }
            return true;
        }
        return Objects.equals(expected, actual);
    }

    @FunctionalInterface
    interface MissingMetadataFieldPolicy {
        boolean allowsMissing(Object key);
    }

    @FunctionalInterface
    interface MetadataValueMismatchPolicy {
        boolean allowsMismatch(Object key, Object expected, Object actual);
    }
}
