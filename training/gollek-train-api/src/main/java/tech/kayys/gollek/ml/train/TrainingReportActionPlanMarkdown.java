package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Objects;

/**
 * Compact Markdown renderer for single-report training action plans.
 */
public final class TrainingReportActionPlanMarkdown {
    private TrainingReportActionPlanMarkdown() {
    }

    public static String render(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return render(report.actionPlan(), report.dataHealth());
    }

    public static String render(TrainingReportActionPlan actionPlan) {
        return render(actionPlan, null);
    }

    private static String render(TrainingReportActionPlan actionPlan, TrainingReportDataHealth dataHealth) {
        Objects.requireNonNull(actionPlan, "actionPlan must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Action Plan");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + actionPlan.status().name() + "`");
        appendLine(markdown, "**Highest diagnostic severity:** `"
                + escapeInline(actionPlan.diagnosticSummary().highestSeverity()) + "`");
        appendLine(markdown, "**Diagnostics:** `" + actionPlan.diagnosticSummary().total() + "` total, `"
                + actionPlan.blockers().size() + "` blockers.");
        appendLine(markdown, "");
        appendDataHealth(markdown, dataHealth);
        appendRecommendations(markdown, actionPlan.recommendations());
        appendActionItems(markdown, actionPlan.actionItems());
        return markdown.toString();
    }

    private static void appendDataHealth(StringBuilder markdown, TrainingReportDataHealth dataHealth) {
        String section = TrainingReportDataHealthMarkdown.render(dataHealth);
        if (!section.isBlank()) {
            markdown.append(section);
        }
    }

    private static void appendRecommendations(
            StringBuilder markdown,
            List<TrainingReportRecommendation> recommendations) {
        appendLine(markdown, "## Recommendations");
        appendLine(markdown, "");
        if (recommendations == null || recommendations.isEmpty()) {
            appendLine(markdown, "No recommendations. The report is ready under the current diagnostics.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown, "| Priority | Category | Diagnostic | Title |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        for (TrainingReportRecommendation recommendation : recommendations) {
            appendLine(markdown, recommendationRow(recommendation));
        }
        appendLine(markdown, "");
    }

    private static void appendActionItems(StringBuilder markdown, List<String> actionItems) {
        if (actionItems == null || actionItems.isEmpty()) {
            return;
        }
        appendLine(markdown, "## Action Items");
        appendLine(markdown, "");
        int index = 1;
        for (String action : actionItems) {
            appendLine(markdown, index + ". " + escapeListItem(action));
            index++;
        }
    }

    private static String recommendationRow(TrainingReportRecommendation recommendation) {
        return "| `" + recommendation.priority().name() + "`"
                + " | `" + recommendation.category().name() + "`"
                + " | `" + escapeTable(recommendation.diagnosticCode()) + "`"
                + " | " + escapeTable(recommendation.title())
                + " |";
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
