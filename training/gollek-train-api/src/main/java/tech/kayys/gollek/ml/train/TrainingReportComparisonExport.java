package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Portable table/JSON export for trainer report comparison results.
 */
public record TrainingReportComparisonExport(
        List<Map<String, Object>> metricRows,
        List<Map<String, Object>> findingRows) {
    public static final String SCHEMA = "gollek.training.report.comparison-export.v1";

    private static final List<String> METRIC_COLUMNS = List.of(
            "metric",
            "direction",
            "available",
            "baseline",
            "candidate",
            "absoluteDelta",
            "relativeDelta",
            "verdict");
    private static final List<String> FINDING_COLUMNS = List.of(
            "severity",
            "code",
            "message",
            "metric",
            "evidence");

    public TrainingReportComparisonExport {
        metricRows = metricRows == null ? List.of() : immutableRows(metricRows);
        findingRows = findingRows == null ? List.of() : immutableRows(findingRows);
    }

    public static TrainingReportComparisonExport fromComparison(TrainingReportComparison comparison) {
        if (comparison == null) {
            return empty();
        }
        Set<String> improved = comparison.improvedMetrics().stream()
                .map(TrainingReportComparison.MetricDelta::name)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
        Set<String> regressed = comparison.regressedMetrics().stream()
                .map(TrainingReportComparison.MetricDelta::name)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());

        List<Map<String, Object>> metrics = comparison.metrics().stream()
                .map(metric -> metricRow(metric, improved, regressed))
                .toList();
        List<Map<String, Object>> findings = comparison.findings().stream()
                .map(TrainingReportComparisonExport::findingRow)
                .toList();
        return new TrainingReportComparisonExport(metrics, findings);
    }

    public static TrainingReportComparisonExport fromMap(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return empty();
        }
        Object schema = map.get("schema");
        if (schema != null && !SCHEMA.equals(String.valueOf(schema))) {
            throw new IllegalArgumentException("Unsupported comparison export schema: " + schema);
        }
        return new TrainingReportComparisonExport(rows(map.get("metrics"), "metrics"), rows(map.get("findings"), "findings"));
    }

    public static TrainingReportComparisonExport empty() {
        return new TrainingReportComparisonExport(List.of(), List.of());
    }

    public boolean available() {
        return !metricRows.isEmpty() || !findingRows.isEmpty();
    }

    public int metricCount() {
        return metricRows.size();
    }

    public int findingCount() {
        return findingRows.size();
    }

    public boolean hasFindings() {
        return !findingRows.isEmpty();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("schema", SCHEMA);
        map.put("metrics", metricRows);
        map.put("findings", findingRows);
        map.put("metricCount", metricCount());
        map.put("findingCount", findingCount());
        map.put("hasFindings", hasFindings());
        return Map.copyOf(map);
    }

    public String toJson() {
        return TrainerJson.toJson(toMap());
    }

    public String metricsCsv() {
        return TrainerCsv.toCsv(METRIC_COLUMNS, metricRows);
    }

    public String findingsCsv() {
        return TrainerCsv.toCsv(FINDING_COLUMNS, findingRows);
    }

    private static Map<String, Object> metricRow(
            TrainingReportComparison.MetricDelta metric,
            Set<String> improved,
            Set<String> regressed) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("metric", metric.name());
        row.put("direction", metric.direction().name());
        row.put("available", metric.available());
        metric.baselineValue().ifPresent(value -> row.put("baseline", value));
        metric.candidateValue().ifPresent(value -> row.put("candidate", value));
        metric.absoluteDelta().ifPresent(value -> row.put("absoluteDelta", value));
        metric.relativeDelta().ifPresent(value -> row.put("relativeDelta", value));
        row.put("verdict", verdict(metric, improved, regressed));
        return Map.copyOf(row);
    }

    private static String verdict(
            TrainingReportComparison.MetricDelta metric,
            Set<String> improved,
            Set<String> regressed) {
        if (!metric.available()) {
            return "UNAVAILABLE";
        }
        if (regressed.contains(metric.name())) {
            return "REGRESSED";
        }
        if (improved.contains(metric.name())) {
            return "IMPROVED";
        }
        return "UNCHANGED";
    }

    private static Map<String, Object> findingRow(TrainingReportDiagnostics.Finding finding) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("severity", finding.severity().name());
        row.put("code", finding.code());
        row.put("message", finding.message());
        Object metric = finding.evidence().get("metric");
        if (metric != null) {
            row.put("metric", metric);
        }
        row.put("evidence", finding.evidence());
        return Map.copyOf(row);
    }

    private static List<Map<String, Object>> immutableRows(List<Map<String, Object>> rows) {
        return rows.stream()
                .map(Map::copyOf)
                .toList();
    }

    private static List<Map<String, Object>> rows(Object value, String fieldName) {
        if (value == null) {
            return List.of();
        }
        if (!(value instanceof Iterable<?> iterable)) {
            throw new IllegalArgumentException("comparison export " + fieldName + " must be an array");
        }
        java.util.ArrayList<Map<String, Object>> rows = new java.util.ArrayList<>();
        for (Object item : iterable) {
            if (!(item instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException("comparison export " + fieldName + " entries must be objects");
            }
            rows.add(stringKeyMap(map));
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> stringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
    }
}
