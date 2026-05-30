package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Publishes recent loss slope diagnostics for trend/plateau monitoring.
 */
final class TrainerLossSlopeMetadata {
    private static final int DEFAULT_WINDOW_SIZE = 5;
    private static final int MIN_POINTS = 3;
    private static final double TREND_EPSILON = 1e-12;

    private TrainerLossSlopeMetadata() {
    }

    static void putLatest(Map<String, Object> metadata, List<Map<String, Object>> rows) {
        putLatest(metadata, rows, "trainLoss", "latestTrainLoss");
        putLatest(metadata, rows, "validationLoss", "latestValidationLoss");
    }

    static void putEpoch(Map<String, Object> row, List<Map<String, Object>> rows, String lossKey) {
        put(row, latestPointsThroughEpoch(row, rows, lossKey), lossKey);
    }

    private static void putLatest(
            Map<String, Object> metadata,
            List<Map<String, Object>> rows,
            String lossKey,
            String prefix) {
        put(metadata, latestPoints(rows, lossKey), prefix);
    }

    private static void put(Map<String, Object> metadata, List<Point> points, String prefix) {
        String availabilityKey = prefix + "SlopeAvailable";
        String slopeKey = prefix + "SlopePerEpoch";
        String trendKey = prefix + "Trend";
        String windowKey = prefix + "SlopeWindowSize";
        boolean available = points.size() >= MIN_POINTS;
        metadata.put(availabilityKey, available);
        if (!available) {
            metadata.remove(slopeKey);
            metadata.remove(trendKey);
            metadata.remove(windowKey);
            return;
        }
        double slope = slope(points);
        metadata.put(slopeKey, slope);
        metadata.put(trendKey, trend(slope));
        metadata.put(windowKey, points.size());
    }

    private static List<Point> latestPoints(List<Map<String, Object>> rows, String lossKey) {
        List<Point> points = new ArrayList<>();
        int observationIndex = 0;
        for (Map<String, Object> row : rows) {
            Double y = finiteNumber(row.get(lossKey));
            if (y == null) {
                continue;
            }
            Double epoch = finiteNumber(row.get("epoch"));
            points.add(new Point(epoch == null ? observationIndex : epoch, y));
            observationIndex++;
        }
        int from = Math.max(0, points.size() - DEFAULT_WINDOW_SIZE);
        return points.subList(from, points.size());
    }

    private static List<Point> latestPointsThroughEpoch(
            Map<String, Object> currentRow,
            List<Map<String, Object>> rows,
            String lossKey) {
        Double currentLoss = finiteNumber(currentRow.get(lossKey));
        if (currentLoss == null) {
            return List.of();
        }
        double currentEpoch = epoch(currentRow.get("epoch"), Double.MAX_VALUE);
        List<Point> points = new ArrayList<>();
        int observationIndex = 0;
        for (Map<String, Object> row : rows) {
            if (!row.containsKey(lossKey)) {
                continue;
            }
            Double y = finiteNumber(row.get(lossKey));
            if (y == null) {
                continue;
            }
            double x = epoch(row.get("epoch"), observationIndex);
            if (x <= currentEpoch) {
                points.add(new Point(x, y));
            }
            observationIndex++;
        }
        if (!rows.contains(currentRow)) {
            points.add(new Point(currentEpoch, currentLoss));
        }
        points.sort(Comparator.comparingDouble(Point::x));
        int from = Math.max(0, points.size() - DEFAULT_WINDOW_SIZE);
        return points.subList(from, points.size());
    }

    private static double epoch(Object value, double fallback) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : fallback;
        }
        return fallback;
    }

    private static double slope(List<Point> points) {
        double meanX = 0.0;
        double meanY = 0.0;
        for (Point point : points) {
            meanX += point.x();
            meanY += point.y();
        }
        meanX /= points.size();
        meanY /= points.size();

        double numerator = 0.0;
        double denominator = 0.0;
        for (Point point : points) {
            double dx = point.x() - meanX;
            numerator += dx * (point.y() - meanY);
            denominator += dx * dx;
        }
        return denominator == 0.0 ? 0.0 : numerator / denominator;
    }

    private static String trend(double slope) {
        if (slope < -TREND_EPSILON) {
            return "improving";
        }
        if (slope > TREND_EPSILON) {
            return "regressing";
        }
        return "flat";
    }

    private static Double finiteNumber(Object value) {
        if (value instanceof Number number) {
            double doubleValue = number.doubleValue();
            return Double.isFinite(doubleValue) ? doubleValue : null;
        }
        return null;
    }

    private record Point(double x, double y) {
    }
}
