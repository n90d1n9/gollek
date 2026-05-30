package tech.kayys.gollek.train.data.multimodal;

import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.train.data.Dataset;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Audits train/validation splits for multimodal training workflows.
 */
public final class MultimodalSplitDiagnostics {
    private MultimodalSplitDiagnostics() {
    }

    public static MultimodalSplitReport inspect(Dataset.Split<List<MultimodalContent>> split) {
        Objects.requireNonNull(split, "split must not be null");
        if (split.train().size() == 0) {
            throw new IllegalArgumentException("train split must not be empty");
        }
        if (split.validation().size() == 0) {
            throw new IllegalArgumentException("validation split must not be empty");
        }

        MultimodalDatasetReport trainReport = MultimodalDatasetDiagnostics.inspect(split.train());
        MultimodalDatasetReport validationReport = MultimodalDatasetDiagnostics.inspect(split.validation());
        return new MultimodalSplitReport(
                trainReport,
                validationReport,
                MultimodalDatasetSplits.overlappingSourcePaths(split),
                shareDelta(
                        trainReport.sampleSignatureCounts(),
                        trainReport.sampleCount(),
                        validationReport.sampleSignatureCounts(),
                        validationReport.sampleCount()),
                shareDelta(
                        trainReport.mimeTypeCounts(),
                        countTotal(trainReport.mimeTypeCounts()),
                        validationReport.mimeTypeCounts(),
                        countTotal(validationReport.mimeTypeCounts())));
    }

    public static Map<String, Double> shareDelta(
            Map<String, Integer> trainCounts,
            int trainTotal,
            Map<String, Integer> validationCounts,
            int validationTotal) {
        Objects.requireNonNull(trainCounts, "trainCounts must not be null");
        Objects.requireNonNull(validationCounts, "validationCounts must not be null");
        if (trainTotal < 0 || validationTotal < 0) {
            throw new IllegalArgumentException("distribution totals must not be negative");
        }

        Set<String> keys = new TreeSet<>();
        keys.addAll(trainCounts.keySet());
        keys.addAll(validationCounts.keySet());
        Map<String, Double> deltas = new LinkedHashMap<>();
        for (String key : keys) {
            String normalized = Objects.requireNonNull(key, "distribution key must not be null").trim();
            if (normalized.isEmpty()) {
                throw new IllegalArgumentException("distribution keys must not be blank");
            }
            double trainShare = share(trainCounts.getOrDefault(key, 0), trainTotal);
            double validationShare = share(validationCounts.getOrDefault(key, 0), validationTotal);
            deltas.put(normalized, Math.abs(trainShare - validationShare));
        }
        return Collections.unmodifiableMap(deltas);
    }

    private static double share(int count, int total) {
        if (count < 0) {
            throw new IllegalArgumentException("distribution counts must not be negative");
        }
        return total == 0 ? 0.0 : (double) count / total;
    }

    private static int countTotal(Map<String, Integer> counts) {
        int total = 0;
        for (int count : counts.values()) {
            if (count < 0) {
                throw new IllegalArgumentException("distribution counts must not be negative");
            }
            total = Math.addExact(total, count);
        }
        return total;
    }
}
