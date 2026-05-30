package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Portable table/JSON export for chart-ready trainer report series.
 */
public record TrainingReportSeriesExport(
        List<String> seriesNames,
        List<Map<String, Object>> rows,
        List<Map<String, Object>> points) {
    public static final String SCHEMA = "gollek.training.report.series-export.v1";

    public TrainingReportSeriesExport {
        seriesNames = seriesNames == null ? List.of() : List.copyOf(seriesNames);
        rows = rows == null ? List.of() : immutableRows(rows);
        points = points == null ? List.of() : immutableRows(points);
    }

    public static TrainingReportSeriesExport fromBundle(TrainingReportSeriesBundle bundle) {
        if (bundle == null) {
            return empty();
        }
        Map<String, TrainingReportSeries> series = new TreeMap<>(bundle.availableSeries());
        if (series.isEmpty()) {
            return empty();
        }

        TreeSet<Integer> epochs = new TreeSet<>();
        List<Map<String, Object>> points = new ArrayList<>();
        for (Map.Entry<String, TrainingReportSeries> entry : series.entrySet()) {
            String name = entry.getKey();
            for (TrainingReportSeries.Point point : entry.getValue().points()) {
                epochs.add(point.epoch());
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("series", name);
                row.put("epoch", point.epoch());
                row.put("value", point.value());
                points.add(Map.copyOf(row));
            }
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        for (Integer epoch : epochs) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("epoch", epoch);
            for (Map.Entry<String, TrainingReportSeries> entry : series.entrySet()) {
                Double value = entry.getValue().toMap().get(epoch);
                if (value != null) {
                    row.put(entry.getKey(), value);
                }
            }
            rows.add(Map.copyOf(row));
        }

        return new TrainingReportSeriesExport(List.copyOf(series.keySet()), rows, points);
    }

    public static TrainingReportSeriesExport empty() {
        return new TrainingReportSeriesExport(List.of(), List.of(), List.of());
    }

    public boolean available() {
        return !seriesNames.isEmpty() && !rows.isEmpty();
    }

    public int epochCount() {
        return rows.size();
    }

    public int pointCount() {
        return points.size();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> export = new LinkedHashMap<>();
        export.put("schema", SCHEMA);
        export.put("seriesNames", seriesNames);
        export.put("rows", rows);
        export.put("points", points);
        export.put("epochCount", epochCount());
        export.put("pointCount", pointCount());
        return Map.copyOf(export);
    }

    public String toJson() {
        return TrainerJson.toJson(toMap());
    }

    public String toCsv() {
        List<String> columns = new ArrayList<>(seriesNames.size() + 1);
        columns.add("epoch");
        columns.addAll(seriesNames);
        return TrainerCsv.toCsv(columns, rows);
    }

    public String toLongCsv() {
        return TrainerCsv.toCsv(List.of("series", "epoch", "value"), points);
    }

    private static List<Map<String, Object>> immutableRows(List<Map<String, Object>> rows) {
        return rows.stream()
                .map(Map::copyOf)
                .toList();
    }
}
