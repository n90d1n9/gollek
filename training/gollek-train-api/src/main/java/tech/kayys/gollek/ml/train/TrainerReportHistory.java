package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Extracts report-friendly epoch history and compact trend summaries.
 */
final class TrainerReportHistory {
    private static final double TREND_EPSILON = 1.0e-12;

    private TrainerReportHistory() {
    }

    static List<Map<String, Object>> rows(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return List.of();
        }
        Object rawRows = metadata.get("epochHistory");
        if (!(rawRows instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object rawRow : iterable) {
            if (rawRow instanceof Map<?, ?> row) {
                rows.add(copyRow(row));
            }
        }
        rows.sort(Comparator.comparingInt(TrainerReportHistory::epochOf));
        return rows.isEmpty() ? List.of() : List.copyOf(rows);
    }

    static Map<String, Object> summary(List<Map<String, Object>> rows) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("available", rows != null && !rows.isEmpty());
        summary.put("size", rows == null ? 0 : rows.size());
        if (rows == null || rows.isEmpty()) {
            summary.put("trainLoss", seriesSummary(List.of(), "trainLoss", true, true));
            summary.put("trainMetrics", Map.of());
            summary.put("validationLoss", seriesSummary(List.of(), "validationLoss", true, true));
            summary.put("validationMetrics", Map.of());
            summary.put("generalization", generalizationSummary(List.of()));
            summary.put("learningRate", seriesSummary(List.of(), "learningRate", false, false));
            summary.put("optimization", TrainerOptimizationHistorySummary.summarize(List.of()));
            return Map.copyOf(summary);
        }
        summary.put("firstEpoch", epochOf(rows.get(0)));
        summary.put("lastEpoch", epochOf(rows.get(rows.size() - 1)));
        summary.put("trainLoss", seriesSummary(rows, "trainLoss", true, true));
        summary.put("trainMetrics", metricSummaries(rows, "trainMetrics", "trainMetric."));
        summary.put("validationLoss", seriesSummary(rows, "validationLoss", true, true));
        summary.put("validationMetrics", metricSummaries(rows, "validationMetrics", "validationMetric."));
        summary.put("generalization", generalizationSummary(rows));
        summary.put("learningRate", seriesSummary(rows, "learningRate", false, false));
        summary.put("optimization", TrainerOptimizationHistorySummary.summarize(rows));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> seriesSummary(
            List<Map<String, Object>> rows,
            String key,
            boolean lowerIsBetter,
            boolean qualityTrend) {
        Map<String, Object> summary = new LinkedHashMap<>();
        List<Point> points = finitePoints(rows, key);
        summary.put("available", !points.isEmpty());
        summary.put("count", points.size());
        if (points.isEmpty()) {
            return Map.copyOf(summary);
        }

        Point first = points.get(0);
        Point latest = points.get(points.size() - 1);
        Point best = best(points, lowerIsBetter);
        double delta = latest.value() - first.value();

        summary.put("first", first.value());
        summary.put("firstEpoch", first.epoch());
        summary.put("latest", latest.value());
        summary.put("latestEpoch", latest.epoch());
        summary.put("best", best.value());
        summary.put("bestEpoch", best.epoch());
        summary.put("deltaFromFirst", delta);
        summary.put("relativeDeltaFromFirst", relativeDelta(first.value(), delta));
        summary.put("trend", trend(delta, lowerIsBetter, qualityTrend));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> generalizationSummary(List<Map<String, Object>> rows) {
        Map<String, Object> summary = new LinkedHashMap<>();
        List<Point> gaps = generalizationPoints(rows);
        summary.put("available", !gaps.isEmpty());
        summary.put("count", gaps.size());
        if (gaps.isEmpty()) {
            return Map.copyOf(summary);
        }

        Point first = gaps.get(0);
        Point latest = gaps.get(gaps.size() - 1);
        Point max = max(gaps);
        double delta = latest.value() - first.value();
        double latestTrainLoss = finiteValueAt(rows, latest.epoch(), "trainLoss");
        double latestValidationLoss = finiteValueAt(rows, latest.epoch(), "validationLoss");

        summary.put("firstGap", first.value());
        summary.put("firstEpoch", first.epoch());
        summary.put("latestGap", latest.value());
        summary.put("latestEpoch", latest.epoch());
        summary.put("maxGap", max.value());
        summary.put("maxGapEpoch", max.epoch());
        summary.put("gapDeltaFromFirst", delta);
        summary.put("gapTrend", gapTrend(delta));
        summary.put("gapIncreasing", delta > TREND_EPSILON);
        summary.put("latestValidationToTrainLossRatio", ratio(latestValidationLoss, latestTrainLoss));
        summary.put("latestValidationLossAboveTrainLoss", latest.value() > TREND_EPSILON);
        return Map.copyOf(summary);
    }

    private static Map<String, Object> metricSummaries(
            List<Map<String, Object>> rows,
            String metricsKey,
            String flatPrefix) {
        Set<String> names = metricNames(rows, metricsKey, flatPrefix);
        if (names.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> summaries = new LinkedHashMap<>();
        for (String name : names) {
            summaries.put(name, metricSeriesSummary(rows, metricsKey, flatPrefix, name));
        }
        return Map.copyOf(summaries);
    }

    private static Map<String, Object> metricSeriesSummary(
            List<Map<String, Object>> rows,
            String metricsKey,
            String flatPrefix,
            String metricName) {
        Map<String, Object> summary = new LinkedHashMap<>();
        List<Point> points = metricPoints(rows, metricsKey, flatPrefix, metricName);
        summary.put("available", !points.isEmpty());
        summary.put("count", points.size());
        if (points.isEmpty()) {
            return Map.copyOf(summary);
        }

        Point first = points.get(0);
        Point latest = points.get(points.size() - 1);
        Point min = min(points);
        Point max = max(points);
        double delta = latest.value() - first.value();

        summary.put("first", first.value());
        summary.put("firstEpoch", first.epoch());
        summary.put("latest", latest.value());
        summary.put("latestEpoch", latest.epoch());
        summary.put("min", min.value());
        summary.put("minEpoch", min.epoch());
        summary.put("max", max.value());
        summary.put("maxEpoch", max.epoch());
        summary.put("deltaFromFirst", delta);
        summary.put("relativeDeltaFromFirst", relativeDelta(first.value(), delta));
        summary.put("trend", trend(delta, false, false));
        return Map.copyOf(summary);
    }

    private static List<Point> finitePoints(List<Map<String, Object>> rows, String key) {
        List<Point> points = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            Object rawValue = row.get(key);
            if (rawValue instanceof Number number) {
                double value = number.doubleValue();
                if (Double.isFinite(value)) {
                    points.add(new Point(epochOf(row), value));
                }
            }
        }
        return points;
    }

    private static List<Point> metricPoints(
            List<Map<String, Object>> rows,
            String metricsKey,
            String flatPrefix,
            String metricName) {
        List<Point> points = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            Object rawValue = metricValue(row, metricsKey, flatPrefix, metricName);
            if (rawValue instanceof Number number) {
                double value = number.doubleValue();
                if (Double.isFinite(value)) {
                    points.add(new Point(epochOf(row), value));
                }
            }
        }
        return points;
    }

    private static List<Point> generalizationPoints(List<Map<String, Object>> rows) {
        List<Point> points = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            Double gap = finiteNumber(row.get("generalizationGap"));
            if (gap == null) {
                Double trainLoss = finiteNumber(row.get("trainLoss"));
                Double validationLoss = finiteNumber(row.get("validationLoss"));
                if (trainLoss != null && validationLoss != null) {
                    gap = validationLoss - trainLoss;
                }
            }
            if (gap != null) {
                points.add(new Point(epochOf(row), gap));
            }
        }
        return points;
    }

    private static Point best(List<Point> points, boolean lowerIsBetter) {
        Point best = points.get(0);
        for (Point point : points) {
            if ((lowerIsBetter && point.value() < best.value())
                    || (!lowerIsBetter && point.value() > best.value())) {
                best = point;
            }
        }
        return best;
    }

    private static Point min(List<Point> points) {
        Point min = points.get(0);
        for (Point point : points) {
            if (point.value() < min.value()) {
                min = point;
            }
        }
        return min;
    }

    private static Point max(List<Point> points) {
        Point max = points.get(0);
        for (Point point : points) {
            if (point.value() > max.value()) {
                max = point;
            }
        }
        return max;
    }

    private static double relativeDelta(double first, double delta) {
        if (Math.abs(first) <= TREND_EPSILON) {
            return Double.NaN;
        }
        return delta / Math.abs(first);
    }

    private static String trend(double delta, boolean lowerIsBetter, boolean qualityTrend) {
        if (Math.abs(delta) <= TREND_EPSILON) {
            return "flat";
        }
        if (!qualityTrend) {
            return delta < 0.0 ? "decreased" : "increased";
        }
        boolean improved = lowerIsBetter ? delta < 0.0 : delta > 0.0;
        return improved ? "improved" : "worsened";
    }

    private static String gapTrend(double delta) {
        if (Math.abs(delta) <= TREND_EPSILON) {
            return "flat";
        }
        return delta > 0.0 ? "increasing" : "decreasing";
    }

    private static Map<String, Object> copyRow(Map<?, ?> row) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : row.entrySet()) {
            copy.put(
                    String.valueOf(entry.getKey()),
                    TrainingReportSnapshots.immutableSnapshot(entry.getValue()));
        }
        return Map.copyOf(copy);
    }

    private static Set<String> metricNames(
            List<Map<String, Object>> rows,
            String metricsKey,
            String flatPrefix) {
        Set<String> names = new TreeSet<>();
        for (Map<String, Object> row : rows) {
            Object rawMetrics = row.get(metricsKey);
            if (rawMetrics instanceof Map<?, ?> metrics) {
                for (Map.Entry<?, ?> entry : metrics.entrySet()) {
                    if (entry.getValue() instanceof Number number && Double.isFinite(number.doubleValue())) {
                        names.add(String.valueOf(entry.getKey()));
                    }
                }
            }
            for (Map.Entry<String, Object> entry : row.entrySet()) {
                if (entry.getKey().startsWith(flatPrefix)
                        && entry.getValue() instanceof Number number
                        && Double.isFinite(number.doubleValue())) {
                    names.add(entry.getKey().substring(flatPrefix.length()));
                }
            }
        }
        return names;
    }

    private static Object metricValue(
            Map<String, Object> row,
            String metricsKey,
            String flatPrefix,
            String metricName) {
        Object rawMetrics = row.get(metricsKey);
        if (rawMetrics instanceof Map<?, ?> metrics && metrics.containsKey(metricName)) {
            return metrics.get(metricName);
        }
        return row.get(flatPrefix + metricName);
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }

    private static double finiteValueAt(List<Map<String, Object>> rows, int epoch, String key) {
        for (Map<String, Object> row : rows) {
            if (epochOf(row) == epoch) {
                Double value = finiteNumber(row.get(key));
                return value == null ? Double.NaN : value;
            }
        }
        return Double.NaN;
    }

    private static double ratio(double numerator, double denominator) {
        if (!Double.isFinite(numerator) || !Double.isFinite(denominator) || Math.abs(denominator) <= TREND_EPSILON) {
            return Double.NaN;
        }
        return numerator / denominator;
    }

    private static int epochOf(Map<String, Object> row) {
        Object epoch = row.get("epoch");
        return epoch instanceof Number number ? number.intValue() : Integer.MAX_VALUE;
    }

    private record Point(int epoch, double value) {
    }
}
