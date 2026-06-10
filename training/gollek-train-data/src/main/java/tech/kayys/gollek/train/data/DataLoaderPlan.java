package tech.kayys.gollek.train.data;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Immutable, serializable view of a loader's runtime batching plan.
 */
public record DataLoaderPlan(
        String kind,
        int datasetSize,
        int sampleCount,
        int batchSize,
        int batchCount,
        boolean sampled,
        boolean shuffle,
        boolean dropLast,
        Long shuffleSeed,
        boolean reshuffleEachEpoch,
        long initialEpoch) {

    public DataLoaderPlan {
        kind = normalizeKind(kind);
        DataLoaderCounts.requireDatasetSize(datasetSize);
        DataLoaderCounts.requireSampleCount(sampleCount);
        DataLoaderBatchSizes.requirePositive(batchSize);
        DataLoaderCounts.requireBatchCount(batchCount);
        DataLoaderEpochs.requireInitialEpoch(initialEpoch);
    }

    public boolean hasShuffleSeed() {
        return shuffleSeed != null;
    }

    public double sampleCoverageRatio() {
        return datasetSize == 0 ? 0.0 : sampleCount / (double) datasetSize;
    }

    public int nominalBatchCapacity() {
        return batchCount * batchSize;
    }

    public String summary() {
        return kind
                + "[datasetSize=" + datasetSize
                + ", sampleCount=" + sampleCount
                + ", batchSize=" + batchSize
                + ", batchCount=" + batchCount
                + ", sampled=" + sampled
                + ", shuffle=" + shuffle
                + ", dropLast=" + dropLast
                + ", reshuffleEachEpoch=" + reshuffleEachEpoch
                + ", initialEpoch=" + initialEpoch
                + (shuffleSeed == null ? "" : ", shuffleSeed=" + shuffleSeed)
                + "]";
    }

    public Map<String, Object> toMetadata(String prefix) {
        String normalizedPrefix = normalizePrefix(prefix);
        Map<String, Object> metadata = new LinkedHashMap<>();
        put(metadata, normalizedPrefix, "kind", kind);
        put(metadata, normalizedPrefix, "datasetSize", datasetSize);
        put(metadata, normalizedPrefix, "sampleCount", sampleCount);
        put(metadata, normalizedPrefix, "batchSize", batchSize);
        put(metadata, normalizedPrefix, "batchCount", batchCount);
        put(metadata, normalizedPrefix, "sampled", sampled);
        put(metadata, normalizedPrefix, "shuffle", shuffle);
        put(metadata, normalizedPrefix, "dropLast", dropLast);
        put(metadata, normalizedPrefix, "hasShuffleSeed", hasShuffleSeed());
        if (shuffleSeed != null) {
            put(metadata, normalizedPrefix, "shuffleSeed", shuffleSeed);
        }
        put(metadata, normalizedPrefix, "reshuffleEachEpoch", reshuffleEachEpoch);
        put(metadata, normalizedPrefix, "initialEpoch", initialEpoch);
        put(metadata, normalizedPrefix, "sampleCoverageRatio", sampleCoverageRatio());
        put(metadata, normalizedPrefix, "nominalBatchCapacity", nominalBatchCapacity());
        return Map.copyOf(metadata);
    }

    private static void put(Map<String, Object> metadata, String prefix, String key, Object value) {
        metadata.put(prefix.isBlank() ? key : prefix + "." + key, value);
    }

    private static String normalizeKind(String kind) {
        String normalized = kind == null ? "" : kind.trim().toLowerCase(Locale.ROOT);
        if (normalized.isBlank()) {
            throw new IllegalArgumentException("kind must not be blank");
        }
        return normalized;
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
