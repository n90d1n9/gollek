package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

/**
 * Human-readable Markdown renderer for training report portfolio exports.
 */
public final class TrainingReportPortfolioMarkdown {
    private static final DecimalFormat NUMBER_FORMAT = new DecimalFormat(
            "0.######",
            DecimalFormatSymbols.getInstance(Locale.ROOT));

    private TrainingReportPortfolioMarkdown() {
    }

    public static String render(TrainingReportPortfolio portfolio) {
        Objects.requireNonNull(portfolio, "portfolio must not be null");
        return render(portfolio.export());
    }

    public static String render(TrainingReportPortfolio portfolio, String baselineName) {
        Objects.requireNonNull(portfolio, "portfolio must not be null");
        return render(portfolio.exportAgainst(baselineName));
    }

    public static String render(TrainingReportPortfolioExport export) {
        Objects.requireNonNull(export, "export must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Portfolio Export");
        appendLine(markdown, "");
        appendLine(markdown, "**Entries:** `" + export.entryCount() + "`");
        appendLine(markdown, "**Comparison metrics:** `" + export.comparisonMetricCount() + "`");
        appendLine(markdown, "**Comparison findings:** `" + export.comparisonFindingCount() + "`");
        appendLine(markdown, "");
        appendLeaderboard(markdown, export.leaderboardRows());
        appendComparisonSummary(markdown, export.comparisonMetricRows(), export.comparisonFindingRows());
        appendComparisonFindings(markdown, export.comparisonFindingRows());
        return markdown.toString();
    }

    public static String render(TrainingReportPortfolioArtifacts.ArtifactBundle bundle) {
        Objects.requireNonNull(bundle, "bundle must not be null");
        StringBuilder markdown = new StringBuilder(render(bundle.export()));
        appendArtifactBundle(markdown, bundle);
        return markdown.toString();
    }

    public static String render(TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Portfolio Artifact Verification");
        appendLine(markdown, "");
        appendLine(markdown, "**Verification:** `" + (verification.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, verification.message());
        appendLine(markdown, "");
        appendArtifactInspection(markdown, verification);
        if (!verification.failures().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Verification Failures");
            appendLine(markdown, "");
            for (String failure : verification.failures()) {
                appendLine(markdown, "- " + escapeListItem(failure));
            }
        }
        return markdown.toString();
    }

    private static void appendLeaderboard(StringBuilder markdown, List<Map<String, Object>> rows) {
        appendLine(markdown, "## Leaderboard");
        appendLine(markdown, "");
        if (rows.isEmpty()) {
            appendLine(markdown, "No reports are available in this portfolio.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown,
                "| Rank | Name | Validation score | Best val loss | Latest val loss | Latest train loss | Epochs | Duration ms | Diagnostics | Data health | Data issues |");
        appendLine(markdown, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |");
        for (Map<String, Object> row : rows) {
            appendLine(markdown, leaderboardRow(row));
        }
        appendLine(markdown, "");
    }

    private static String leaderboardRow(Map<String, Object> row) {
        return "| " + value(row, "rank", "-")
                + " | `" + escapeTable(value(row, "name", "unnamed")) + "`"
                + " | " + number(row.get("validationScore"))
                + " | " + number(row.get("bestValidationLoss"))
                + " | " + number(row.get("latestValidationLoss"))
                + " | " + number(row.get("latestTrainLoss"))
                + " | " + value(row, "epochCount", "-")
                + " | " + value(row, "durationMs", "-")
                + " | `" + escapeTable(value(row, "highestDiagnosticSeverity", "UNKNOWN")) + "`"
                + " | `" + escapeTable(value(row, "dataHealthStatus", "not-recorded")) + "`"
                + " | " + value(row, "dataHealthIssueCount", "0")
                + " |";
    }

    private static void appendComparisonSummary(
            StringBuilder markdown,
            List<Map<String, Object>> metricRows,
            List<Map<String, Object>> findingRows) {
        if (metricRows.isEmpty() && findingRows.isEmpty()) {
            return;
        }
        Map<String, CandidateSummary> summaries = comparisonSummaries(metricRows, findingRows);
        appendLine(markdown, "## Baseline Comparison Summary");
        appendLine(markdown, "");
        appendLine(markdown, "| Candidate | Improved | Regressed | Unchanged | Unavailable | Findings |");
        appendLine(markdown, "| --- | ---: | ---: | ---: | ---: | ---: |");
        for (CandidateSummary summary : summaries.values()) {
            appendLine(markdown, "| `" + escapeTable(summary.candidate) + "`"
                    + " | " + summary.improved
                    + " | " + summary.regressed
                    + " | " + summary.unchanged
                    + " | " + summary.unavailable
                    + " | " + summary.findings
                    + " |");
        }
        appendLine(markdown, "");
    }

    private static void appendComparisonFindings(
            StringBuilder markdown,
            List<Map<String, Object>> findingRows) {
        if (findingRows.isEmpty()) {
            return;
        }
        appendLine(markdown, "## Comparison Findings");
        appendLine(markdown, "");
        appendLine(markdown, "| Candidate | Severity | Code | Metric | Message |");
        appendLine(markdown, "| --- | --- | --- | --- | --- |");
        for (Map<String, Object> row : findingRows) {
            appendLine(markdown, "| `" + escapeTable(value(row, "candidateReport", "candidate")) + "`"
                    + " | `" + escapeTable(value(row, "severity", "UNKNOWN")) + "`"
                    + " | `" + escapeTable(value(row, "code", "unknown")) + "`"
                    + " | `" + escapeTable(value(row, "metric", "-")) + "`"
                    + " | " + escapeTable(value(row, "message", "-"))
                    + " |");
        }
        appendLine(markdown, "");
    }

    private static void appendArtifactBundle(
            StringBuilder markdown,
            TrainingReportPortfolioArtifacts.ArtifactBundle bundle) {
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 |");
        appendLine(markdown, "| --- | --- | --- |");
        appendLine(markdown, artifactRow("JSON export", bundle.jsonFile(), bundle.jsonSha256()));
        appendLine(markdown, artifactRow("Markdown summary", bundle.markdownFile(), bundle.markdownSha256()));
        appendLine(markdown, artifactRow(
                "Leaderboard CSV",
                bundle.leaderboardCsvFile(),
                bundle.leaderboardCsvSha256()));
        appendLine(markdown, artifactRow(
                "Comparison metrics CSV",
                bundle.comparisonMetricsCsvFile(),
                bundle.comparisonMetricsCsvSha256()));
        appendLine(markdown, artifactRow(
                "Comparison findings CSV",
                bundle.comparisonFindingsCsvFile(),
                bundle.comparisonFindingsCsvSha256()));
    }

    private static void appendArtifactInspection(
            StringBuilder markdown,
            TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
        TrainingReportPortfolioArtifacts.ArtifactInspection inspection = verification.inspection();
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 | Verified |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        appendLine(markdown, artifactVerificationRow(
                "JSON export",
                inspection.jsonFile(),
                inspection.jsonSha256(),
                verification.jsonSha256Matches()));
        appendLine(markdown, artifactVerificationRow(
                "Markdown summary",
                inspection.markdownFile(),
                inspection.markdownSha256(),
                verification.markdownSha256Matches()));
        appendLine(markdown, artifactVerificationRow(
                "Leaderboard CSV",
                inspection.leaderboardCsvFile(),
                inspection.leaderboardCsvSha256(),
                verification.leaderboardCsvSha256Matches()));
        appendLine(markdown, artifactVerificationRow(
                "Comparison metrics CSV",
                inspection.comparisonMetricsCsvFile(),
                inspection.comparisonMetricsCsvSha256(),
                verification.comparisonMetricsCsvSha256Matches()));
        appendLine(markdown, artifactVerificationRow(
                "Comparison findings CSV",
                inspection.comparisonFindingsCsvFile(),
                inspection.comparisonFindingsCsvSha256(),
                verification.comparisonFindingsCsvSha256Matches()));
    }

    private static Map<String, CandidateSummary> comparisonSummaries(
            List<Map<String, Object>> metricRows,
            List<Map<String, Object>> findingRows) {
        Map<String, CandidateSummary> summaries = new LinkedHashMap<>();
        for (Map<String, Object> row : metricRows) {
            String candidate = value(row, "candidateReport", "candidate");
            CandidateSummary summary = summaries.computeIfAbsent(candidate, CandidateSummary::new);
            switch (value(row, "verdict", "UNKNOWN")) {
                case "IMPROVED" -> summary.improved++;
                case "REGRESSED" -> summary.regressed++;
                case "UNCHANGED" -> summary.unchanged++;
                case "UNAVAILABLE" -> summary.unavailable++;
                default -> {
                    // Unknown verdicts are intentionally ignored to keep the renderer forward-compatible.
                }
            }
        }
        for (Map<String, Object> row : findingRows) {
            String candidate = value(row, "candidateReport", "candidate");
            summaries.computeIfAbsent(candidate, CandidateSummary::new).findings++;
        }
        return summaries;
    }

    private static String artifactRow(String label, Path path, String sha256) {
        return "| " + escapeTable(label)
                + " | `" + escapeTable(path.toString()) + "`"
                + " | `" + escapeTable(shortSha(sha256)) + "` |";
    }

    private static String artifactVerificationRow(
            String label,
            Path path,
            String sha256,
            boolean verified) {
        return "| " + escapeTable(label)
                + " | `" + escapeTable(path.toString()) + "`"
                + " | `" + escapeTable(shortSha(sha256)) + "`"
                + " | `" + (verified ? "yes" : "no") + "` |";
    }

    private static String number(Object value) {
        OptionalDouble parsed = optionalDouble(value);
        return parsed.isPresent() ? formatNumber(parsed.getAsDouble()) : "n/a";
    }

    private static String value(Map<String, Object> row, String key, String fallback) {
        return stringValue(row.get(key), fallback);
    }

    private static String formatNumber(double value) {
        if (!Double.isFinite(value)) {
            return "n/a";
        }
        synchronized (NUMBER_FORMAT) {
            return NUMBER_FORMAT.format(value);
        }
    }

    private static String shortSha(String sha256) {
        if (sha256 == null || sha256.isBlank()) {
            return "n/a";
        }
        String normalized = sha256.trim();
        return normalized.length() <= 12 ? normalized : normalized.substring(0, 12);
    }

    private static String escapeListItem(String value) {
        return escapeInline(value).replace("\n", " ");
    }

    private static String escapeTable(String value) {
        return escapeInline(value).replace("|", "\\|").replace("\n", " ");
    }

    private static String escapeInline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }

    private static final class CandidateSummary {
        private final String candidate;
        private int improved;
        private int regressed;
        private int unchanged;
        private int unavailable;
        private int findings;

        private CandidateSummary(String candidate) {
            this.candidate = candidate;
        }
    }
}
