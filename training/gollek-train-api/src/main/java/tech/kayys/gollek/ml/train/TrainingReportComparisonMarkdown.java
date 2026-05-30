package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.optionalDouble;
import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * Human-readable Markdown renderer for baseline-vs-candidate trainer comparisons.
 */
public final class TrainingReportComparisonMarkdown {
    private static final DecimalFormat NUMBER_FORMAT = new DecimalFormat(
            "0.######",
            DecimalFormatSymbols.getInstance(Locale.ROOT));

    private TrainingReportComparisonMarkdown() {
    }

    public static String render(TrainingReportComparison comparison) {
        Objects.requireNonNull(comparison, "comparison must not be null");
        return render(comparison.export());
    }

    public static String render(TrainingReportComparisonExport export) {
        Objects.requireNonNull(export, "export must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Report Comparison");
        appendLine(markdown, "");
        appendLine(markdown, "**Metrics:** `" + export.metricCount() + "`");
        appendLine(markdown, "**Findings:** `" + export.findingCount() + "`");
        appendLine(markdown, "**Status:** `" + (export.hasFindings() ? "ATTENTION" : "PASS") + "`");
        appendLine(markdown, "");
        appendVerdictSummary(markdown, export.metricRows());
        appendMetricRows(markdown, export.metricRows());
        appendFindingRows(markdown, export.findingRows());
        return markdown.toString();
    }

    public static String render(TrainingReportComparisonArtifacts.ArtifactBundle bundle) {
        Objects.requireNonNull(bundle, "bundle must not be null");
        StringBuilder markdown = new StringBuilder(render(bundle.export()));
        appendArtifactBundle(markdown, bundle);
        return markdown.toString();
    }

    public static String render(TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Comparison Artifact Verification");
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

    private static void appendVerdictSummary(StringBuilder markdown, List<Map<String, Object>> rows) {
        int improved = 0;
        int regressed = 0;
        int unchanged = 0;
        int unavailable = 0;
        for (Map<String, Object> row : rows) {
            switch (value(row, "verdict", "UNKNOWN")) {
                case "IMPROVED" -> improved++;
                case "REGRESSED" -> regressed++;
                case "UNCHANGED" -> unchanged++;
                case "UNAVAILABLE" -> unavailable++;
                default -> {
                    // Unknown verdicts are intentionally ignored for forward compatibility.
                }
            }
        }
        appendLine(markdown, "## Summary");
        appendLine(markdown, "");
        appendLine(markdown, "| Improved | Regressed | Unchanged | Unavailable |");
        appendLine(markdown, "| ---: | ---: | ---: | ---: |");
        appendLine(markdown, "| " + improved + " | " + regressed + " | " + unchanged + " | " + unavailable + " |");
        appendLine(markdown, "");
    }

    private static void appendMetricRows(StringBuilder markdown, List<Map<String, Object>> rows) {
        appendLine(markdown, "## Metrics");
        appendLine(markdown, "");
        if (rows.isEmpty()) {
            appendLine(markdown, "No comparable metrics are available.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown, "| Metric | Direction | Baseline | Candidate | Delta | Relative Delta | Verdict |");
        appendLine(markdown, "| --- | --- | ---: | ---: | ---: | ---: | --- |");
        for (Map<String, Object> row : rows) {
            appendLine(markdown, "| `" + escapeTable(value(row, "metric", "unknown")) + "`"
                    + " | `" + escapeTable(value(row, "direction", "NEUTRAL")) + "`"
                    + " | " + number(row.get("baseline"))
                    + " | " + number(row.get("candidate"))
                    + " | " + number(row.get("absoluteDelta"))
                    + " | " + percent(row.get("relativeDelta"))
                    + " | `" + escapeTable(value(row, "verdict", "UNKNOWN")) + "`"
                    + " |");
        }
        appendLine(markdown, "");
    }

    private static void appendFindingRows(StringBuilder markdown, List<Map<String, Object>> rows) {
        appendLine(markdown, "## Findings");
        appendLine(markdown, "");
        if (rows.isEmpty()) {
            appendLine(markdown, "No regression findings were detected.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown, "| Severity | Code | Metric | Message |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        for (Map<String, Object> row : rows) {
            appendLine(markdown, "| `" + escapeTable(value(row, "severity", "UNKNOWN")) + "`"
                    + " | `" + escapeTable(value(row, "code", "unknown")) + "`"
                    + " | `" + escapeTable(value(row, "metric", "-")) + "`"
                    + " | " + escapeTable(value(row, "message", "-"))
                    + " |");
        }
        appendLine(markdown, "");
    }

    private static void appendArtifactBundle(
            StringBuilder markdown,
            TrainingReportComparisonArtifacts.ArtifactBundle bundle) {
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 |");
        appendLine(markdown, "| --- | --- | --- |");
        appendLine(markdown, artifactRow("JSON export", bundle.jsonFile(), bundle.jsonSha256()));
        appendLine(markdown, artifactRow("Markdown report", bundle.markdownFile(), bundle.markdownSha256()));
        appendLine(markdown, artifactRow("JUnit XML report", bundle.junitXmlFile(), bundle.junitXmlSha256()));
        appendLine(markdown, artifactRow("Metrics CSV", bundle.metricsCsvFile(), bundle.metricsCsvSha256()));
        appendLine(markdown, artifactRow("Findings CSV", bundle.findingsCsvFile(), bundle.findingsCsvSha256()));
    }

    private static void appendArtifactInspection(
            StringBuilder markdown,
            TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        TrainingReportComparisonArtifacts.ArtifactInspection inspection = verification.inspection();
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
                "Markdown report",
                inspection.markdownFile(),
                inspection.markdownSha256(),
                verification.markdownSha256Matches() && verification.markdownMatchesJson()));
        appendLine(markdown, artifactVerificationRow(
                "JUnit XML report",
                inspection.junitXmlFile(),
                inspection.junitXmlSha256(),
                verification.junitXmlSha256Matches()
                        && verification.junitXmlWellFormed()
                        && verification.junitXmlMatchesJson()));
        appendLine(markdown, artifactVerificationRow(
                "Metrics CSV",
                inspection.metricsCsvFile(),
                inspection.metricsCsvSha256(),
                verification.metricsCsvSha256Matches() && verification.metricsCsvMatchesJson()));
        appendLine(markdown, artifactVerificationRow(
                "Findings CSV",
                inspection.findingsCsvFile(),
                inspection.findingsCsvSha256(),
                verification.findingsCsvSha256Matches() && verification.findingsCsvMatchesJson()));
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

    private static String percent(Object value) {
        OptionalDouble parsed = optionalDouble(value);
        return parsed.isPresent() ? formatNumber(parsed.getAsDouble() * 100.0) + "%" : "n/a";
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
}
