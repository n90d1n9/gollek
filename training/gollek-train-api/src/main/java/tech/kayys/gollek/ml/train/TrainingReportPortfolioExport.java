package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Portable leaderboard and comparison-table export for multi-run training portfolios.
 */
public record TrainingReportPortfolioExport(
        List<Map<String, Object>> leaderboardRows,
        List<Map<String, Object>> comparisonMetricRows,
        List<Map<String, Object>> comparisonFindingRows) {
    public static final String SCHEMA = "gollek.training.report.portfolio-export.v1";

    private static final List<String> LEADERBOARD_COLUMNS = List.of(
            "rank",
            "name",
            "validationScore",
            "bestValidationLoss",
            "latestValidationLoss",
            "latestTrainLoss",
            "epochCount",
            "durationMs",
            "highestDiagnosticSeverity",
            "dataHealthStatus",
            "dataHealthAvailable",
            "dataHealthGatePassed",
            "dataHealthIssueCount",
            "dataHealthWarningCount",
            "dataHealthErrorCount",
            "dataHealthIssueCodes",
            "source",
            "sourceBytes",
            "sourceSha256");
    private static final List<String> COMPARISON_METRIC_COLUMNS = List.of(
            "baselineReport",
            "candidateReport",
            "metric",
            "direction",
            "available",
            "baseline",
            "candidate",
            "absoluteDelta",
            "relativeDelta",
            "verdict");
    private static final List<String> COMPARISON_FINDING_COLUMNS = List.of(
            "baselineReport",
            "candidateReport",
            "severity",
            "code",
            "message",
            "metric",
            "evidence");

    public TrainingReportPortfolioExport {
        leaderboardRows = leaderboardRows == null ? List.of() : immutableRows(leaderboardRows);
        comparisonMetricRows = comparisonMetricRows == null ? List.of() : immutableRows(comparisonMetricRows);
        comparisonFindingRows = comparisonFindingRows == null ? List.of() : immutableRows(comparisonFindingRows);
    }

    public static TrainingReportPortfolioExport fromPortfolio(TrainingReportPortfolio portfolio) {
        if (portfolio == null) {
            return empty();
        }
        return new TrainingReportPortfolioExport(leaderboardRows(portfolio), List.of(), List.of());
    }

    public static TrainingReportPortfolioExport fromPortfolio(
            TrainingReportPortfolio portfolio,
            String baselineName) {
        if (portfolio == null) {
            return empty();
        }
        TrainingReportPortfolio.Entry baseline = portfolio.entry(baselineName)
                .orElseThrow(() -> new IllegalArgumentException("Unknown baseline report: " + baselineName));
        List<Map<String, Object>> comparisonMetrics = new ArrayList<>();
        List<Map<String, Object>> comparisonFindings = new ArrayList<>();
        for (TrainingReportPortfolio.Entry candidate : portfolio.rankedByValidationScore()) {
            if (candidate.name().equals(baseline.name())) {
                continue;
            }
            TrainingReportComparisonExport comparisonExport =
                    TrainingReportComparison.compare(baseline.report(), candidate.report()).export();
            for (Map<String, Object> row : comparisonExport.metricRows()) {
                comparisonMetrics.add(comparisonRow(baseline.name(), candidate.name(), row));
            }
            for (Map<String, Object> row : comparisonExport.findingRows()) {
                comparisonFindings.add(comparisonRow(baseline.name(), candidate.name(), row));
            }
        }
        return new TrainingReportPortfolioExport(
                leaderboardRows(portfolio),
                comparisonMetrics,
                comparisonFindings);
    }

    public static TrainingReportPortfolioExport empty() {
        return new TrainingReportPortfolioExport(List.of(), List.of(), List.of());
    }

    public static TrainingReportPortfolioExport fromMap(Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return empty();
        }
        Object schema = map.get("schema");
        if (schema != null && !SCHEMA.equals(String.valueOf(schema))) {
            throw new IllegalArgumentException("Unsupported portfolio export schema: " + schema);
        }
        return new TrainingReportPortfolioExport(
                rows(map.get("leaderboard"), "leaderboard"),
                rows(map.get("comparisonMetrics"), "comparisonMetrics"),
                rows(map.get("comparisonFindings"), "comparisonFindings"));
    }

    public boolean available() {
        return !leaderboardRows.isEmpty();
    }

    public int entryCount() {
        return leaderboardRows.size();
    }

    public int comparisonMetricCount() {
        return comparisonMetricRows.size();
    }

    public int comparisonFindingCount() {
        return comparisonFindingRows.size();
    }

    public boolean hasComparisons() {
        return !comparisonMetricRows.isEmpty();
    }

    public boolean hasComparisonFindings() {
        return !comparisonFindingRows.isEmpty();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("schema", SCHEMA);
        map.put("leaderboard", leaderboardRows);
        map.put("comparisonMetrics", comparisonMetricRows);
        map.put("comparisonFindings", comparisonFindingRows);
        map.put("entryCount", entryCount());
        map.put("comparisonMetricCount", comparisonMetricCount());
        map.put("comparisonFindingCount", comparisonFindingCount());
        map.put("hasComparisons", hasComparisons());
        map.put("hasComparisonFindings", hasComparisonFindings());
        return Map.copyOf(map);
    }

    public String toJson() {
        return TrainerJson.toJson(toMap());
    }

    public String leaderboardCsv() {
        return TrainerCsv.toCsv(LEADERBOARD_COLUMNS, leaderboardRows);
    }

    public String comparisonMetricsCsv() {
        return TrainerCsv.toCsv(COMPARISON_METRIC_COLUMNS, comparisonMetricRows);
    }

    public String comparisonFindingsCsv() {
        return TrainerCsv.toCsv(COMPARISON_FINDING_COLUMNS, comparisonFindingRows);
    }

    private static List<Map<String, Object>> leaderboardRows(TrainingReportPortfolio portfolio) {
        List<Map<String, Object>> rows = new ArrayList<>();
        int rank = 1;
        for (TrainingReportPortfolio.Entry entry : portfolio.rankedByValidationScore()) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("rank", rank++);
            row.put("name", entry.name());
            putOptional(row, "validationScore", entry.validationScore());
            putOptional(row, "bestValidationLoss", entry.report().bestValidationLoss());
            putOptional(row, "latestValidationLoss", entry.report().latestValidationLoss());
            putOptional(row, "latestTrainLoss", entry.report().latestTrainLoss());
            row.put("epochCount", entry.report().epochCount());
            row.put("durationMs", entry.report().durationMs());
            row.put("highestDiagnosticSeverity", entry.report().highestDiagnosticSeverity());
            appendDataHealth(row, entry.report().dataHealth());
            if (entry.source() != null) {
                row.put("source", entry.source().toString());
            }
            if (entry.sourceBytes() != null) {
                row.put("sourceBytes", entry.sourceBytes());
            }
            if (entry.sourceSha256() != null) {
                row.put("sourceSha256", entry.sourceSha256());
            }
            rows.add(Map.copyOf(row));
        }
        return List.copyOf(rows);
    }

    private static void appendDataHealth(Map<String, Object> row, TrainingReportDataHealth dataHealth) {
        TrainingReportDataHealthSummary summary = TrainingReportDataHealthSummary.from(dataHealth);
        row.put("dataHealthStatus", summary.status());
        row.put("dataHealthAvailable", summary.available());
        row.put("dataHealthGatePassed", summary.gatePassed());
        row.put("dataHealthIssueCount", summary.issueCount());
        row.put("dataHealthWarningCount", summary.warningCount());
        row.put("dataHealthErrorCount", summary.errorCount());
        row.put("dataHealthIssueCodes", String.join("|", summary.issueCodes()));
    }

    private static Map<String, Object> comparisonRow(
            String baselineName,
            String candidateName,
            Map<String, Object> source) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("baselineReport", baselineName);
        row.put("candidateReport", candidateName);
        row.putAll(source);
        return Map.copyOf(row);
    }

    private static void putOptional(Map<String, Object> target, String key, OptionalDouble value) {
        if (value != null && value.isPresent()) {
            target.put(key, value.getAsDouble());
        }
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
            throw new IllegalArgumentException("portfolio export " + fieldName + " must be an array");
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object item : iterable) {
            if (!(item instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException("portfolio export " + fieldName + " entries must be objects");
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
