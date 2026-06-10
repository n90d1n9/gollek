package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Compact report summary for optimizer telemetry across epoch history.
 */
final class TrainerOptimizationHistorySummary {
    private static final double TREND_EPSILON = 1.0e-12;

    private TrainerOptimizationHistorySummary() {
    }

    static Map<String, Object> summarize(List<Map<String, Object>> rows) {
        List<Map<String, Object>> telemetryRows = telemetryRows(rows);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("available", !telemetryRows.isEmpty());
        summary.put("count", telemetryRows.size());
        summary.put("gradients", gradientSummary(telemetryRows));
        summary.put("parameters", parameterSummary(telemetryRows));
        summary.put("parameterUpdates", parameterUpdateSummary(telemetryRows));
        summary.put("latest", latestSummary(telemetryRows));
        if (!telemetryRows.isEmpty()) {
            summary.put("firstEpoch", epochOf(telemetryRows.get(0)));
            summary.put("lastEpoch", epochOf(telemetryRows.get(telemetryRows.size() - 1)));
        }
        return Map.copyOf(summary);
    }

    private static Map<String, Object> gradientSummary(List<Map<String, Object>> rows) {
        if (rows.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("l2Norm", seriesSummary(rows, "gradientL2Norm"));
        summary.put("zeroFraction", seriesSummary(rows, "gradientZeroFraction"));
        summary.put("nonFiniteFraction", seriesSummary(rows, "gradientNonFiniteFraction"));
        summary.put("nonFiniteCount", totalSummary(rows, "gradientNonFiniteCount"));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> parameterSummary(List<Map<String, Object>> rows) {
        if (rows.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("l2Norm", seriesSummary(rows, "parameterL2Norm"));
        summary.put("zeroFraction", seriesSummary(rows, "parameterZeroFraction"));
        summary.put("nonFiniteFraction", seriesSummary(rows, "parameterNonFiniteFraction"));
        summary.put("nonFiniteCount", totalSummary(rows, "parameterNonFiniteCount"));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> parameterUpdateSummary(List<Map<String, Object>> rows) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("enabled", anyTrue(rows, "parameterUpdateDiagnosticsEnabled"));
        if (rows.isEmpty()) {
            return Map.copyOf(summary);
        }
        summary.put("l2Norm", seriesSummary(rows, "parameterUpdateL2Norm"));
        summary.put("zeroFraction", seriesSummary(rows, "parameterUpdateZeroFraction"));
        summary.put("nonFiniteFraction", seriesSummary(rows, "parameterUpdateNonFiniteFraction"));
        summary.put("nonFiniteCount", totalSummary(rows, "parameterUpdateNonFiniteCount"));
        summary.put("toParameterL2Ratio", seriesSummary(rows, "parameterUpdateToParameterL2Ratio"));
        summary.put("toGradientL2Ratio", seriesSummary(rows, "parameterUpdateToGradientL2Ratio"));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> latestSummary(List<Map<String, Object>> rows) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("available", !rows.isEmpty());
        if (rows.isEmpty()) {
            return Map.copyOf(summary);
        }
        Map<String, Object> latest = rows.get(rows.size() - 1);
        summary.put("epoch", epochOf(latest));
        putIfFinite(summary, "gradientL2Norm", latest.get("gradientL2Norm"));
        putIfFinite(summary, "gradientZeroFraction", latest.get("gradientZeroFraction"));
        putIfPresent(summary, "gradientNonFiniteCount", latest.get("gradientNonFiniteCount"));
        putIfFinite(summary, "parameterL2Norm", latest.get("parameterL2Norm"));
        putIfPresent(summary, "parameterNonFiniteCount", latest.get("parameterNonFiniteCount"));
        putIfPresent(summary, "parameterUpdateDiagnosticsEnabled", latest.get("parameterUpdateDiagnosticsEnabled"));
        putIfFinite(summary, "parameterUpdateL2Norm", latest.get("parameterUpdateL2Norm"));
        putIfFinite(summary, "parameterUpdateToParameterL2Ratio", latest.get("parameterUpdateToParameterL2Ratio"));
        putIfPresent(summary, "parameterUpdateNonFiniteCount", latest.get("parameterUpdateNonFiniteCount"));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> seriesSummary(List<Map<String, Object>> rows, String key) {
        Map<String, Object> summary = new LinkedHashMap<>();
        List<Point> points = finitePoints(rows, key);
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
        summary.put("trend", trend(delta));
        return Map.copyOf(summary);
    }

    private static Map<String, Object> totalSummary(List<Map<String, Object>> rows, String key) {
        Map<String, Object> summary = new LinkedHashMap<>();
        long total = 0L;
        long max = 0L;
        int maxEpoch = Integer.MAX_VALUE;
        int count = 0;
        for (Map<String, Object> row : rows) {
            Long value = longValue(row.get(key));
            if (value == null) {
                continue;
            }
            count++;
            total += value.longValue();
            if (value.longValue() > max) {
                max = value.longValue();
                maxEpoch = epochOf(row);
            }
        }
        summary.put("available", count > 0);
        summary.put("count", count);
        summary.put("total", total);
        summary.put("max", max);
        if (count > 0) {
            summary.put("maxEpoch", maxEpoch);
        }
        return Map.copyOf(summary);
    }

    private static List<Map<String, Object>> telemetryRows(List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) {
            return List.of();
        }
        List<Map<String, Object>> telemetryRows = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            if (hasTelemetry(row)) {
                telemetryRows.add(row);
            }
        }
        return telemetryRows.isEmpty() ? List.of() : List.copyOf(telemetryRows);
    }

    private static boolean hasTelemetry(Map<String, Object> row) {
        return row.containsKey("gradientL2Norm")
                || row.containsKey("gradientZeroFraction")
                || row.containsKey("gradientNonFiniteCount")
                || row.containsKey("parameterL2Norm")
                || row.containsKey("parameterNonFiniteCount")
                || row.containsKey("parameterUpdateDiagnosticsEnabled")
                || row.containsKey("parameterUpdateL2Norm")
                || row.containsKey("parameterUpdateToParameterL2Ratio");
    }

    private static List<Point> finitePoints(List<Map<String, Object>> rows, String key) {
        List<Point> points = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            Double value = finiteNumber(row.get(key));
            if (value != null) {
                points.add(new Point(epochOf(row), value.doubleValue()));
            }
        }
        return points;
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

    private static String trend(double delta) {
        if (Math.abs(delta) <= TREND_EPSILON) {
            return "flat";
        }
        return delta < 0.0 ? "decreased" : "increased";
    }

    private static void putIfFinite(Map<String, Object> target, String key, Object value) {
        Double finite = finiteNumber(value);
        if (finite != null) {
            target.put(key, finite);
        }
    }

    private static void putIfPresent(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, TrainingReportSnapshots.immutableSnapshot(value));
        }
    }

    private static boolean anyTrue(List<Map<String, Object>> rows, String key) {
        for (Map<String, Object> row : rows) {
            Object value = row.get(key);
            if (value instanceof Boolean bool && bool.booleanValue()) {
                return true;
            }
            if (value instanceof String text && Boolean.parseBoolean(text)) {
                return true;
            }
        }
        return false;
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        if (value instanceof String text) {
            try {
                double doubleValue = Double.parseDouble(text);
                return Double.isFinite(doubleValue) ? doubleValue : null;
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static Long longValue(Object value) {
        if (value instanceof Number number) {
            return Math.max(0L, number.longValue());
        }
        if (value instanceof String text) {
            try {
                return Math.max(0L, Long.parseLong(text));
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static int epochOf(Map<String, Object> row) {
        Object epoch = row.get("epoch");
        return epoch instanceof Number number ? number.intValue() : Integer.MAX_VALUE;
    }

    private record Point(int epoch, double value) {
    }
}
