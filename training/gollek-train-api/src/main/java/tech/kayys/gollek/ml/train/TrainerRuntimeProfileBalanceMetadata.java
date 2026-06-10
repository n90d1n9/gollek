package tech.kayys.gollek.ml.train;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Builds compact runtime-balance metadata from detailed trainer phase timings.
 */
final class TrainerRuntimeProfileBalanceMetadata {
    private static final String INPUT_GROUP = "input";
    private static final String TRAIN_GROUP = "train";
    private static final String VALIDATION_GROUP = "validation";
    private static final String OPTIMIZER_GROUP = "optimizer";

    private TrainerRuntimeProfileBalanceMetadata() {
    }

    static Map<String, Object> from(
            String prefix,
            Map<TrainerRuntimeProfiler.Phase, TrainerRuntimeProfiler.PhaseSnapshot> snapshots) {
        Objects.requireNonNull(snapshots, "snapshots must not be null");
        String normalizedPrefix = prefix == null || prefix.isBlank() ? "runtimeProfile" : prefix.trim();
        Map<String, Long> groupTotals = groupTotals(snapshots);
        long totalNanos = totalNanos(groupTotals);
        long inputNanos = groupTotals.getOrDefault(INPUT_GROUP, 0L);
        long trainNanos = groupTotals.getOrDefault(TRAIN_GROUP, 0L);
        long validationNanos = groupTotals.getOrDefault(VALIDATION_GROUP, 0L);
        long optimizerNanos = groupTotals.getOrDefault(OPTIMIZER_GROUP, 0L);
        long computeNanos = Math.addExact(Math.addExact(trainNanos, validationNanos), optimizerNanos);

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(normalizedPrefix + ".totalMillis", millis(totalNanos));
        putBucket(metadata, normalizedPrefix, "input", inputNanos, totalNanos);
        putBucket(metadata, normalizedPrefix, "compute", computeNanos, totalNanos);
        putBucket(metadata, normalizedPrefix, TRAIN_GROUP, trainNanos, totalNanos);
        putBucket(metadata, normalizedPrefix, VALIDATION_GROUP, validationNanos, totalNanos);
        putBucket(metadata, normalizedPrefix, OPTIMIZER_GROUP, optimizerNanos, totalNanos);
        putBottleneck(metadata, normalizedPrefix, groupTotals, totalNanos);
        return Map.copyOf(metadata);
    }

    private static Map<String, Long> groupTotals(
            Map<TrainerRuntimeProfiler.Phase, TrainerRuntimeProfiler.PhaseSnapshot> snapshots) {
        Map<String, Long> totals = new LinkedHashMap<>();
        for (Map.Entry<TrainerRuntimeProfiler.Phase, TrainerRuntimeProfiler.PhaseSnapshot> entry : snapshots.entrySet()) {
            totals.merge(entry.getKey().group(), entry.getValue().totalNanos(), Math::addExact);
        }
        return totals;
    }

    private static long totalNanos(Map<String, Long> groupTotals) {
        long total = 0L;
        for (long value : groupTotals.values()) {
            total = Math.addExact(total, value);
        }
        return total;
    }

    private static void putBucket(
            Map<String, Object> metadata,
            String prefix,
            String bucket,
            long nanos,
            long totalNanos) {
        String key = prefix + ".balance." + bucket;
        metadata.put(key + ".totalMillis", millis(nanos));
        metadata.put(key + ".percentTotal", percent(nanos, totalNanos));
    }

    private static void putBottleneck(
            Map<String, Object> metadata,
            String prefix,
            Map<String, Long> groupTotals,
            long totalNanos) {
        groupTotals.entrySet().stream()
                .max(Comparator
                        .<Map.Entry<String, Long>>comparingLong(Map.Entry::getValue)
                        .thenComparing(Map.Entry::getKey))
                .ifPresentOrElse(entry -> {
                    metadata.put(prefix + ".balance.bottleneckGroup", entry.getKey());
                    metadata.put(prefix + ".balance.bottleneck.totalMillis", millis(entry.getValue()));
                    metadata.put(prefix + ".balance.bottleneck.percentTotal", percent(entry.getValue(), totalNanos));
                }, () -> {
                    metadata.put(prefix + ".balance.bottleneckGroup", "none");
                    metadata.put(prefix + ".balance.bottleneck.totalMillis", 0.0);
                    metadata.put(prefix + ".balance.bottleneck.percentTotal", 0.0);
                });
    }

    private static double millis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static double percent(long nanos, long totalNanos) {
        return totalNanos <= 0L ? 0.0 : (nanos * 100.0) / totalNanos;
    }
}
