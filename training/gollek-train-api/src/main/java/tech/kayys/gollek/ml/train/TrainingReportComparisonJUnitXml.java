package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.stringValue;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * JUnit XML renderer for baseline-vs-candidate trainer comparisons.
 */
public final class TrainingReportComparisonJUnitXml {
    private TrainingReportComparisonJUnitXml() {
    }

    public static String render(TrainingReportComparison comparison) {
        Objects.requireNonNull(comparison, "comparison must not be null");
        return render(comparison.export());
    }

    public static String render(TrainingReportComparisonExport export) {
        Objects.requireNonNull(export, "export must not be null");
        String markdown = TrainingReportComparisonMarkdown.render(export);
        int tests = Math.max(1, export.findingCount());
        int failures = export.findingCount();
        VerdictCounts verdicts = verdictCounts(export.metricRows());

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.report.comparison\" tests=\"" + tests
                + "\" failures=\"" + failures + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "comparison.metricCount", Integer.toString(export.metricCount()));
        property(xml, "comparison.findingCount", Integer.toString(export.findingCount()));
        property(xml, "comparison.hasFindings", Boolean.toString(export.hasFindings()));
        property(xml, "comparison.improved", Integer.toString(verdicts.improved()));
        property(xml, "comparison.regressed", Integer.toString(verdicts.regressed()));
        property(xml, "comparison.unchanged", Integer.toString(verdicts.unchanged()));
        property(xml, "comparison.unavailable", Integer.toString(verdicts.unavailable()));
        appendLine(xml, "  </properties>");
        if (export.findingRows().isEmpty()) {
            appendLine(xml, "  <testcase classname=\"gollek.training.report.comparison\""
                    + " name=\"comparison passed\" time=\"0\">");
            appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
            appendLine(xml, "  </testcase>");
        } else {
            for (Map<String, Object> finding : export.findingRows()) {
                appendFindingTestcase(xml, finding);
            }
        }
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    public static String render(TrainingReportComparisonArtifacts.ArtifactBundle bundle) {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return render(bundle.export());
    }

    public static String render(TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.report.comparison.artifacts\" tests=\"10\" failures=\""
                + artifactFailureCount(verification) + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "artifacts.passed", Boolean.toString(verification.passed()));
        property(xml, "artifacts.directory", verification.inspection().directory().toString());
        property(xml, "artifacts.failures", Integer.toString(verification.failures().size()));
        appendLine(xml, "  </properties>");
        appendArtifactCheck(xml, "json checksum", verification.jsonSha256Matches(),
                "JSON checksum mismatch for " + verification.inspection().jsonFile());
        appendArtifactCheck(xml, "markdown checksum", verification.markdownSha256Matches(),
                "Markdown checksum mismatch for " + verification.inspection().markdownFile());
        appendArtifactCheck(xml, "junit xml checksum", verification.junitXmlSha256Matches(),
                "JUnit XML checksum mismatch for " + verification.inspection().junitXmlFile());
        appendArtifactCheck(xml, "metrics csv checksum", verification.metricsCsvSha256Matches(),
                "Metrics CSV checksum mismatch for " + verification.inspection().metricsCsvFile());
        appendArtifactCheck(xml, "findings csv checksum", verification.findingsCsvSha256Matches(),
                "Findings CSV checksum mismatch for " + verification.inspection().findingsCsvFile());
        appendArtifactCheck(xml, "junit xml well formed", verification.junitXmlWellFormed(),
                "JUnit XML is not well-formed: " + verification.inspection().junitXmlFile());
        appendArtifactCheck(xml, "markdown matches json", verification.markdownMatchesJson(),
                "Markdown report does not match JSON export: " + verification.inspection().markdownFile());
        appendArtifactCheck(xml, "junit xml matches json", verification.junitXmlMatchesJson(),
                "JUnit XML report does not match JSON export: " + verification.inspection().junitXmlFile());
        appendArtifactCheck(xml, "metrics csv matches json", verification.metricsCsvMatchesJson(),
                "Metrics CSV does not match JSON export: " + verification.inspection().metricsCsvFile());
        appendArtifactCheck(xml, "findings csv matches json", verification.findingsCsvMatchesJson(),
                "Findings CSV does not match JSON export: " + verification.inspection().findingsCsvFile());
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static void appendFindingTestcase(StringBuilder xml, Map<String, Object> finding) {
        String code = value(finding, "code", "comparison.finding");
        String severity = value(finding, "severity", "UNKNOWN");
        String message = value(finding, "message", code);
        appendLine(xml, "  <testcase classname=\"gollek.training.report.comparison\" name=\""
                + escapeXml(code) + "\" time=\"0\">");
        appendLine(xml, "    <failure type=\"" + escapeXml(severity)
                + "\" message=\"" + escapeXml(message) + "\">");
        appendLine(xml, escapeText(TrainerJson.toJson(finding)));
        appendLine(xml, "    </failure>");
        appendLine(xml, "    <system-out>" + escapeText(message) + "</system-out>");
        appendLine(xml, "  </testcase>");
    }

    private static void appendArtifactCheck(
            StringBuilder xml,
            String name,
            boolean passed,
            String failureMessage) {
        appendLine(xml, "  <testcase classname=\"gollek.training.report.comparison.artifacts\" name=\""
                + escapeXml(name) + "\" time=\"0\">");
        if (!passed) {
            appendLine(xml, "    <failure type=\"ARTIFACT_VERIFICATION\" message=\""
                    + escapeXml(failureMessage) + "\">");
            appendLine(xml, escapeText(failureMessage));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "  </testcase>");
    }

    private static int artifactFailureCount(TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        int failures = 0;
        failures += verification.jsonSha256Matches() ? 0 : 1;
        failures += verification.markdownSha256Matches() ? 0 : 1;
        failures += verification.junitXmlSha256Matches() ? 0 : 1;
        failures += verification.metricsCsvSha256Matches() ? 0 : 1;
        failures += verification.findingsCsvSha256Matches() ? 0 : 1;
        failures += verification.junitXmlWellFormed() ? 0 : 1;
        failures += verification.markdownMatchesJson() ? 0 : 1;
        failures += verification.junitXmlMatchesJson() ? 0 : 1;
        failures += verification.metricsCsvMatchesJson() ? 0 : 1;
        failures += verification.findingsCsvMatchesJson() ? 0 : 1;
        return failures;
    }

    private static VerdictCounts verdictCounts(List<Map<String, Object>> rows) {
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
                    // Unknown verdicts are ignored so future enum values do not break rendering.
                }
            }
        }
        return new VerdictCounts(improved, regressed, unchanged, unavailable);
    }

    private static String value(Map<String, Object> row, String key, String fallback) {
        return stringValue(row.get(key), fallback);
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static String escapeXml(String value) {
        return escapeText(value).replace("\"", "&quot;").replace("'", "&apos;");
    }

    private static String escapeText(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            switch (ch) {
                case '&' -> escaped.append("&amp;");
                case '<' -> escaped.append("&lt;");
                case '>' -> escaped.append("&gt;");
                default -> {
                    if (isValidXmlChar(ch)) {
                        escaped.append(ch);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static boolean isValidXmlChar(char ch) {
        return ch == 0x9
                || ch == 0xA
                || ch == 0xD
                || (ch >= 0x20 && ch <= 0xD7FF)
                || (ch >= 0xE000 && ch <= 0xFFFD);
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }

    private record VerdictCounts(int improved, int regressed, int unchanged, int unavailable) {
    }
}
