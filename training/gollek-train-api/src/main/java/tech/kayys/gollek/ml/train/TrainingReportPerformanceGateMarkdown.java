package tech.kayys.gollek.ml.train;

/**
 * Markdown renderer for trainer performance gate results.
 */
final class TrainingReportPerformanceGateMarkdown {
    private TrainingReportPerformanceGateMarkdown() {
    }

    static String render(TrainingReportPerformanceGate.Result result) {
        if (result == null) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Trainer Performance Gate");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + (result.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Available:** `" + yesNo(result.available()) + "`");
        appendLine(markdown, "**Message:** " + escapeInline(result.message()));
        appendLine(markdown, "");
        if (result.findings().isEmpty()) {
            appendLine(markdown, "No performance gate findings.");
            appendLine(markdown, "");
            return markdown.toString();
        }
        appendLine(markdown, "| Severity | Code | Message | Action |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        for (TrainingReportPerformanceGate.Finding finding : result.findings()) {
            appendLine(markdown, "| `" + escapeTable(finding.severity()) + "`"
                    + " | `" + escapeTable(finding.code()) + "`"
                    + " | " + escapeTable(finding.message())
                    + " | " + escapeTable(finding.action())
                    + " |");
        }
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static String escapeInline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`");
    }

    private static String escapeTable(String value) {
        return escapeInline(value).replace("|", "\\|").replace("\n", " ");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
