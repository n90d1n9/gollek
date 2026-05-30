package tech.kayys.gollek.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Aggregate result for tensor-native cross-validation training runs.
 */
public record CrossValidationTrainingSummary(
        int foldCount,
        List<Fold> folds,
        LossStats latestTrainLoss,
        LossStats latestValidationLoss,
        LossStats bestValidationLoss,
        Map<String, MetricStats> latestTrainMetrics,
        Map<String, MetricStats> latestValidationMetrics,
        Map<String, Object> metadata) {

    public CrossValidationTrainingSummary {
        folds = List.copyOf(folds == null ? List.of() : folds);
        foldCount = foldCount <= 0 ? folds.size() : foldCount;
        metadata = Collections.unmodifiableMap(new LinkedHashMap<>(
                metadata == null ? Map.of() : metadata));
        latestTrainLoss = latestTrainLoss == null
                ? LossStats.from(folds.stream()
                        .map(Fold::summary)
                        .map(TrainingSummary::latestTrainLoss)
                        .toList())
                : latestTrainLoss;
        latestValidationLoss = latestValidationLoss == null
                ? LossStats.from(folds.stream()
                        .map(Fold::summary)
                        .map(TrainingSummary::latestValidationLoss)
                        .toList())
                : latestValidationLoss;
        bestValidationLoss = bestValidationLoss == null
                ? LossStats.from(folds.stream()
                        .map(Fold::summary)
                        .map(TrainingSummary::bestValidationLoss)
                        .toList())
                : bestValidationLoss;
        latestTrainMetrics = latestTrainMetrics == null
                ? aggregateMetricStats(folds, "latestTrainMetrics")
                : immutableMetricStats(latestTrainMetrics);
        latestValidationMetrics = latestValidationMetrics == null
                ? aggregateMetricStats(folds, "latestValidationMetrics")
                : immutableMetricStats(latestValidationMetrics);
    }

    public Optional<Fold> bestFoldByValidationLoss() {
        return folds.stream()
                .filter(fold -> isFinite(fold.summary().bestValidationLoss()))
                .min(Comparator.comparingDouble(fold -> fold.summary().bestValidationLoss()));
    }

    public List<Fold> foldsRankedByValidationLoss() {
        return folds.stream()
                .filter(fold -> isFinite(fold.summary().bestValidationLoss()))
                .sorted(Comparator.comparingDouble(fold -> fold.summary().bestValidationLoss()))
                .toList();
    }

    public Optional<Fold> bestFoldByValidationMetric(String metricName, boolean higherIsBetter) {
        String normalized = requireMetricName(metricName);
        Comparator<Fold> comparator = Comparator.comparingDouble(
                fold -> metricValue(fold, "latestValidationMetrics", normalized).orElseThrow());
        return folds.stream()
                .filter(fold -> metricValue(fold, "latestValidationMetrics", normalized).isPresent())
                .max(higherIsBetter ? comparator : comparator.reversed());
    }

    public List<Fold> foldsRankedByValidationMetric(String metricName, boolean higherIsBetter) {
        String normalized = requireMetricName(metricName);
        Comparator<Fold> comparator = Comparator.comparingDouble(
                fold -> metricValue(fold, "latestValidationMetrics", normalized).orElseThrow());
        if (higherIsBetter) {
            comparator = comparator.reversed();
        }
        return folds.stream()
                .filter(fold -> metricValue(fold, "latestValidationMetrics", normalized).isPresent())
                .sorted(comparator)
                .toList();
    }

    public double meanLatestValidationLoss() {
        return latestValidationLoss.mean();
    }

    public double meanBestValidationLoss() {
        return bestValidationLoss.mean();
    }

    public double meanLatestTrainMetric(String metricName) {
        MetricStats stats = latestTrainMetrics.get(requireMetricName(metricName));
        return stats == null ? Double.NaN : stats.mean();
    }

    public double meanLatestValidationMetric(String metricName) {
        MetricStats stats = latestValidationMetrics.get(requireMetricName(metricName));
        return stats == null ? Double.NaN : stats.mean();
    }

    public List<FoldReport> foldReports() {
        return folds.stream().map(CrossValidationTrainingSummary::foldReport).toList();
    }

    public List<Map<String, Object>> foldReportRows() {
        return foldReports().stream().map(FoldReport::toMap).toList();
    }

    public record Fold(
            int foldIndex,
            int foldCount,
            int trainSize,
            int validationSize,
            TrainingSummary summary) {

        public Fold {
            if (foldIndex < 0) {
                throw new IllegalArgumentException("foldIndex must be >= 0");
            }
            if (foldCount <= 0) {
                throw new IllegalArgumentException("foldCount must be > 0");
            }
            if (trainSize < 0) {
                throw new IllegalArgumentException("trainSize must be >= 0");
            }
            if (validationSize < 0) {
                throw new IllegalArgumentException("validationSize must be >= 0");
            }
            if (summary == null) {
                throw new IllegalArgumentException("summary must not be null");
            }
        }
    }

    public record FoldReport(
            int foldIndex,
            int foldNumber,
            int foldCount,
            int trainSize,
            int validationSize,
            Double latestTrainLoss,
            Double latestValidationLoss,
            double bestValidationLoss,
            int bestValidationEpoch,
            Map<String, Double> latestTrainMetrics,
            Map<String, Double> latestValidationMetrics) {

        public FoldReport {
            if (foldIndex < 0) {
                throw new IllegalArgumentException("foldIndex must be >= 0");
            }
            if (foldNumber <= 0) {
                throw new IllegalArgumentException("foldNumber must be > 0");
            }
            if (foldCount <= 0) {
                throw new IllegalArgumentException("foldCount must be > 0");
            }
            if (trainSize < 0) {
                throw new IllegalArgumentException("trainSize must be >= 0");
            }
            if (validationSize < 0) {
                throw new IllegalArgumentException("validationSize must be >= 0");
            }
            latestTrainMetrics = Collections.unmodifiableMap(new LinkedHashMap<>(
                    latestTrainMetrics == null ? Map.of() : latestTrainMetrics));
            latestValidationMetrics = Collections.unmodifiableMap(new LinkedHashMap<>(
                    latestValidationMetrics == null ? Map.of() : latestValidationMetrics));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("foldIndex", foldIndex);
            row.put("foldNumber", foldNumber);
            row.put("foldCount", foldCount);
            row.put("trainSize", trainSize);
            row.put("validationSize", validationSize);
            row.put("latestTrainLoss", latestTrainLoss);
            row.put("latestValidationLoss", latestValidationLoss);
            row.put("bestValidationLoss", bestValidationLoss);
            row.put("bestValidationEpoch", bestValidationEpoch);
            row.put("latestTrainMetrics", latestTrainMetrics);
            row.put("latestValidationMetrics", latestValidationMetrics);
            return Collections.unmodifiableMap(row);
        }
    }

    public record LossStats(
            double mean,
            double standardDeviation,
            double min,
            double max,
            int finiteCount) {

        static LossStats from(List<Double> values) {
            List<Double> finite = new ArrayList<>();
            for (Double value : values == null ? List.<Double>of() : values) {
                if (value != null && isFinite(value)) {
                    finite.add(value);
                }
            }
            if (finite.isEmpty()) {
                return new LossStats(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0);
            }

            double sum = 0.0;
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (double value : finite) {
                sum += value;
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            double mean = sum / finite.size();
            double variance = 0.0;
            for (double value : finite) {
                double delta = value - mean;
                variance += delta * delta;
            }
            return new LossStats(
                    mean,
                    Math.sqrt(variance / finite.size()),
                    min,
                    max,
                    finite.size());
        }
    }

    public record MetricStats(
            double mean,
            double standardDeviation,
            double min,
            double max,
            int finiteCount) {

        static MetricStats from(List<Double> values) {
            List<Double> finite = finiteValues(values);
            if (finite.isEmpty()) {
                return new MetricStats(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 0);
            }
            ScalarStats stats = scalarStats(finite);
            return new MetricStats(
                    stats.mean(),
                    stats.standardDeviation(),
                    stats.min(),
                    stats.max(),
                    finite.size());
        }
    }

    private record ScalarStats(
            double mean,
            double standardDeviation,
            double min,
            double max) {
    }

    private static Map<String, MetricStats> aggregateMetricStats(List<Fold> folds, String metadataKey) {
        Map<String, List<Double>> valuesByMetric = new LinkedHashMap<>();
        for (Fold fold : folds) {
            Object raw = fold.summary().metadata().get(metadataKey);
            if (!(raw instanceof Map<?, ?> metrics)) {
                continue;
            }
            for (Map.Entry<?, ?> entry : metrics.entrySet()) {
                if (entry.getKey() instanceof String name && entry.getValue() instanceof Number value) {
                    valuesByMetric.computeIfAbsent(name, ignored -> new ArrayList<>())
                            .add(value.doubleValue());
                }
            }
        }

        Map<String, MetricStats> stats = new LinkedHashMap<>();
        for (Map.Entry<String, List<Double>> entry : valuesByMetric.entrySet()) {
            stats.put(entry.getKey(), MetricStats.from(entry.getValue()));
        }
        return Collections.unmodifiableMap(stats);
    }

    private static Map<String, MetricStats> immutableMetricStats(Map<String, MetricStats> stats) {
        Map<String, MetricStats> copy = new LinkedHashMap<>();
        for (Map.Entry<String, MetricStats> entry : stats.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                copy.put(entry.getKey(), entry.getValue());
            }
        }
        return Collections.unmodifiableMap(copy);
    }

    private static OptionalDouble metricValue(Fold fold, String metadataKey, String metricName) {
        Object raw = fold.summary().metadata().get(metadataKey);
        if (!(raw instanceof Map<?, ?> metrics)) {
            return OptionalDouble.empty();
        }
        Object value = metrics.get(metricName);
        if (!(value instanceof Number number) || !isFinite(number.doubleValue())) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(number.doubleValue());
    }

    private static FoldReport foldReport(Fold fold) {
        TrainingSummary summary = fold.summary();
        return new FoldReport(
                fold.foldIndex(),
                fold.foldIndex() + 1,
                fold.foldCount(),
                fold.trainSize(),
                fold.validationSize(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss(),
                summary.bestValidationLoss(),
                summary.bestValidationEpoch(),
                metricMap(summary, "latestTrainMetrics"),
                metricMap(summary, "latestValidationMetrics"));
    }

    private static Map<String, Double> metricMap(TrainingSummary summary, String metadataKey) {
        Object raw = summary.metadata().get(metadataKey);
        if (!(raw instanceof Map<?, ?> metrics)) {
            return Map.of();
        }
        Map<String, Double> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : metrics.entrySet()) {
            if (entry.getKey() instanceof String key && entry.getValue() instanceof Number number) {
                values.put(key, number.doubleValue());
            }
        }
        return values;
    }

    private static String requireMetricName(String metricName) {
        if (metricName == null || metricName.isBlank()) {
            throw new IllegalArgumentException("metricName must not be blank");
        }
        return metricName.trim();
    }

    private static List<Double> finiteValues(List<Double> values) {
        List<Double> finite = new ArrayList<>();
        for (Double value : values == null ? List.<Double>of() : values) {
            if (value != null && isFinite(value)) {
                finite.add(value);
            }
        }
        return finite;
    }

    private static ScalarStats scalarStats(List<Double> finite) {
        double sum = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double value : finite) {
            sum += value;
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        double mean = sum / finite.size();
        double variance = 0.0;
        for (double value : finite) {
            double delta = value - mean;
            variance += delta * delta;
        }
        return new ScalarStats(mean, Math.sqrt(variance / finite.size()), min, max);
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }
}
