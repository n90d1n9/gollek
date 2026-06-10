package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Map;

final class TrainingReportHealthIssueMarkdown {
    private TrainingReportHealthIssueMarkdown() {
    }

    static String renderDetails(String title, List<Map<String, Object>> issues) {
        if (issues == null || issues.isEmpty()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "### " + escapeInline(title));
        appendLine(markdown, "");
        appendLine(markdown, "| Code | Severity | Artifact | Blocking | Message | Evidence |");
        appendLine(markdown, "| --- | --- | --- | --- | --- | --- |");
        for (Map<String, Object> issue : issues) {
            appendLine(markdown, "| "
                    + code(stringValue(issue.get("code")))
                    + " | `" + escapeTable(stringValue(issue.get("severity"))) + "`"
                    + " | `" + escapeTable(stringValue(issue.get("artifact"))) + "`"
                    + " | `" + yesNo(booleanValue(issue.get("blocking"))) + "`"
                    + " | " + escapeTable(stringValue(issue.get("message")))
                    + " | " + escapeTable(TrainingReportEvidenceSummary.compact(issue.get("evidence")))
                    + " |");
        }
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String code(String value) {
        return "`" + escapeInline(value) + "`";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static boolean booleanValue(Object value) {
        return value instanceof Boolean bool && bool;
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
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
