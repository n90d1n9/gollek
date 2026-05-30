package tech.kayys.gollek.ml.train;

import java.util.List;

/**
 * Compact Markdown renderer for trainer data-health report sections.
 */
public final class TrainingReportDataHealthMarkdown {
    private TrainingReportDataHealthMarkdown() {
    }

    public static boolean visible(TrainingReportDataHealth dataHealth) {
        return dataHealth != null
                && (dataHealth.available() || dataHealth.issueDetected() || !dataHealth.gatePassed());
    }

    public static String render(TrainingReportDataHealth dataHealth) {
        if (!visible(dataHealth)) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Data Health");
        appendLine(markdown, "");
        appendLine(markdown, "| Section | Available | Status | Gate | Issues | Warnings | Errors |");
        appendLine(markdown, "| --- | --- | --- | --- | ---: | ---: | ---: |");
        appendLine(markdown, row("Loader plan", dataHealth.loaderPlan()));
        appendLine(markdown, row("Distribution", dataHealth.distribution()));
        appendLine(markdown, "");
        appendIssues(markdown, dataHealth.issueCodes());
        appendActions(markdown, dataHealth.recommendedActions());
        return markdown.toString();
    }

    private static String row(String section, TrainingReportHealthStatus status) {
        return "| " + escapeTable(section)
                + " | `" + yesNo(status.available()) + "`"
                + " | `" + escapeTable(status.status()) + "`"
                + " | `" + (status.gatePassed() ? "PASS" : "FAIL") + "`"
                + " | " + status.issueCount()
                + " | " + status.warningCount()
                + " | " + status.errorCount()
                + " |";
    }

    private static void appendIssues(StringBuilder markdown, List<String> issueCodes) {
        if (issueCodes == null || issueCodes.isEmpty()) {
            return;
        }
        appendLine(markdown, "**Data health issues:** "
                + String.join(", ", issueCodes.stream().map(TrainingReportDataHealthMarkdown::code).toList()));
        appendLine(markdown, "");
    }

    private static void appendActions(StringBuilder markdown, List<String> actions) {
        if (actions == null || actions.isEmpty()) {
            return;
        }
        appendLine(markdown, "### Data Health Actions");
        appendLine(markdown, "");
        int index = 1;
        for (String action : actions) {
            appendLine(markdown, index + ". " + escapeListItem(action));
            index++;
        }
        appendLine(markdown, "");
    }

    private static String code(String value) {
        return "`" + escapeInline(value) + "`";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
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
