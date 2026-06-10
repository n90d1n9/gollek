package tech.kayys.gollek.ml.data;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Serializable description of bounded asynchronous prefetching for ML data loaders.
 */
public record DataLoaderPrefetchPlan(
        boolean enabled,
        int bufferSize,
        int workerCount) {
    public static final int DEFAULT_BUFFER_SIZE = 2;

    public DataLoaderPrefetchPlan {
        if (enabled) {
            DataLoaderBatchSizes.requirePositive(bufferSize);
            if (workerCount <= 0) {
                throw new IllegalArgumentException("workerCount must be positive when prefetching is enabled, got: "
                        + workerCount);
            }
        } else {
            bufferSize = 0;
            workerCount = 0;
        }
    }

    public static DataLoaderPrefetchPlan disabled() {
        return new DataLoaderPrefetchPlan(false, 0, 0);
    }

    public static DataLoaderPrefetchPlan enabled(int bufferSize) {
        return enabled(bufferSize, 1);
    }

    public static DataLoaderPrefetchPlan enabled(int bufferSize, int workerCount) {
        return new DataLoaderPrefetchPlan(true, bufferSize, workerCount);
    }

    public static DataLoaderPrefetchPlan recommended() {
        return enabled(DEFAULT_BUFFER_SIZE);
    }

    public int maxBufferedItems() {
        return bufferSize * workerCount;
    }

    public String summary() {
        if (!enabled) {
            return "prefetch[enabled=false]";
        }
        return "prefetch[enabled=true, bufferSize=" + bufferSize
                + ", workerCount=" + workerCount
                + ", maxBufferedItems=" + maxBufferedItems()
                + "]";
    }

    public Map<String, Object> toMetadata(String prefix) {
        String normalizedPrefix = normalizePrefix(prefix);
        Map<String, Object> metadata = new LinkedHashMap<>();
        put(metadata, normalizedPrefix, "enabled", enabled);
        put(metadata, normalizedPrefix, "bufferSize", bufferSize);
        put(metadata, normalizedPrefix, "workerCount", workerCount);
        put(metadata, normalizedPrefix, "maxBufferedItems", maxBufferedItems());
        put(metadata, normalizedPrefix, "summary", summary());
        return Map.copyOf(metadata);
    }

    private static void put(Map<String, Object> metadata, String prefix, String key, Object value) {
        metadata.put(prefix.isBlank() ? key : prefix + "." + key, value);
    }

    private static String normalizePrefix(String prefix) {
        if (prefix == null || prefix.isBlank()) {
            return "";
        }
        String normalized = prefix.trim();
        while (normalized.endsWith(".")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
    }
}
