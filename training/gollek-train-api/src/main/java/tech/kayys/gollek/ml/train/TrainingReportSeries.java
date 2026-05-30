package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.normalizedString;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Chart-ready numeric series extracted from canonical report epoch snapshots.
 */
public record TrainingReportSeries(String name, List<Point> points) {
    private static final double TREND_EPSILON = 1.0e-12;

    public TrainingReportSeries {
        name = normalizedString(name, "series");
        points = points == null ? List.of() : sanitize(points);
    }

    public static TrainingReportSeries empty(String name) {
        return new TrainingReportSeries(name, List.of());
    }

    public static TrainingReportSeries fromSnapshots(
            String name,
            List<TrainingReportEpochSnapshot> snapshots,
            ValueExtractor extractor) {
        if (snapshots == null || snapshots.isEmpty() || extractor == null) {
            return empty(name);
        }
        java.util.ArrayList<Point> points = new java.util.ArrayList<>();
        for (int index = 0; index < snapshots.size(); index++) {
            TrainingReportEpochSnapshot snapshot = snapshots.get(index);
            OptionalDouble value = extractor.value(snapshot);
            if (value.isPresent()) {
                int epoch = snapshot.epoch().isPresent() ? snapshot.epoch().getAsInt() : index;
                points.add(new Point(epoch, value.getAsDouble()));
            }
        }
        return new TrainingReportSeries(name, points);
    }

    public boolean available() {
        return !points.isEmpty();
    }

    public int count() {
        return points.size();
    }

    public OptionalDouble first() {
        return points.isEmpty() ? OptionalDouble.empty() : OptionalDouble.of(points.get(0).value());
    }

    public OptionalInt firstEpoch() {
        return points.isEmpty() ? OptionalInt.empty() : OptionalInt.of(points.get(0).epoch());
    }

    public OptionalDouble latest() {
        return points.isEmpty()
                ? OptionalDouble.empty()
                : OptionalDouble.of(points.get(points.size() - 1).value());
    }

    public OptionalInt latestEpoch() {
        return points.isEmpty()
                ? OptionalInt.empty()
                : OptionalInt.of(points.get(points.size() - 1).epoch());
    }

    public OptionalDouble min() {
        Point point = minPoint();
        return point == null ? OptionalDouble.empty() : OptionalDouble.of(point.value());
    }

    public OptionalInt minEpoch() {
        Point point = minPoint();
        return point == null ? OptionalInt.empty() : OptionalInt.of(point.epoch());
    }

    public OptionalDouble max() {
        Point point = maxPoint();
        return point == null ? OptionalDouble.empty() : OptionalDouble.of(point.value());
    }

    public OptionalInt maxEpoch() {
        Point point = maxPoint();
        return point == null ? OptionalInt.empty() : OptionalInt.of(point.epoch());
    }

    public OptionalDouble deltaFromFirst() {
        if (points.size() < 2) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(points.get(points.size() - 1).value() - points.get(0).value());
    }

    public OptionalDouble relativeDeltaFromFirst() {
        OptionalDouble delta = deltaFromFirst();
        if (delta.isEmpty() || points.isEmpty()) {
            return OptionalDouble.empty();
        }
        double first = points.get(0).value();
        if (Math.abs(first) <= TREND_EPSILON) {
            return OptionalDouble.empty();
        }
        return OptionalDouble.of(delta.getAsDouble() / Math.abs(first));
    }

    public String trend() {
        OptionalDouble delta = deltaFromFirst();
        if (delta.isEmpty()) {
            return "unknown";
        }
        double value = delta.getAsDouble();
        if (Math.abs(value) <= TREND_EPSILON) {
            return "flat";
        }
        return value > 0.0d ? "increased" : "decreased";
    }

    public boolean increased() {
        return "increased".equals(trend());
    }

    public boolean decreased() {
        return "decreased".equals(trend());
    }

    public boolean flat() {
        return "flat".equals(trend());
    }

    public List<Double> values() {
        return points.stream().map(Point::value).toList();
    }

    public Map<Integer, Double> toMap() {
        Map<Integer, Double> values = new LinkedHashMap<>();
        for (Point point : points) {
            values.put(point.epoch(), point.value());
        }
        return Map.copyOf(values);
    }

    private Point minPoint() {
        if (points.isEmpty()) {
            return null;
        }
        Point min = points.get(0);
        for (Point point : points) {
            if (point.value() < min.value()) {
                min = point;
            }
        }
        return min;
    }

    private Point maxPoint() {
        if (points.isEmpty()) {
            return null;
        }
        Point max = points.get(0);
        for (Point point : points) {
            if (point.value() > max.value()) {
                max = point;
            }
        }
        return max;
    }

    private static List<Point> sanitize(List<Point> points) {
        return points.stream()
                .filter(point -> point != null && Double.isFinite(point.value()))
                .toList();
    }

    @FunctionalInterface
    public interface ValueExtractor {
        OptionalDouble value(TrainingReportEpochSnapshot snapshot);
    }

    public record Point(int epoch, double value) {
        public Point {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("series point value must be finite");
            }
        }
    }
}
