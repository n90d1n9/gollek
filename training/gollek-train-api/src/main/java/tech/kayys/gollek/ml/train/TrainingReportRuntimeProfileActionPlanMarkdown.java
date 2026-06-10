package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Objects;

/**
 * Markdown renderer for prioritized trainer runtime-performance action plans.
 */
public final class TrainingReportRuntimeProfileActionPlanMarkdown {
    private TrainingReportRuntimeProfileActionPlanMarkdown() {
    }

    public static String render(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return render(report.runtimeProfileActionPlan());
    }

    public static String render(TrainingReportRuntimeProfileActionPlan plan) {
        Objects.requireNonNull(plan, "plan must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Runtime Profile Action Plan");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + plan.status().name() + "`");
        appendLine(markdown, "**Targets:** `" + plan.targets().size() + "`");
        appendLine(markdown, "");
        if (!plan.available()) {
            appendLine(markdown, "No runtime profile metadata is available in this report.");
            appendLine(markdown, "");
            return markdown.toString();
        }
        appendTargets(markdown, plan.targets());
        appendActions(markdown, plan.nextActions());
        return markdown.toString();
    }

    private static void appendTargets(
            StringBuilder markdown,
            List<TrainingReportRuntimeProfileActionPlan.Target> targets) {
        appendLine(markdown, "## Runtime Targets");
        appendLine(markdown, "");
        if (targets.isEmpty()) {
            appendLine(markdown, "No runtime tuning targets were detected.");
            appendLine(markdown, "");
            return;
        }
        appendLine(markdown, "| Rank | Kind | Target | Priority | Category | Total ms | Total % | Diagnostic |");
        appendLine(markdown, "| ---: | --- | --- | --- | --- | ---: | ---: | --- |");
        int rank = 1;
        for (TrainingReportRuntimeProfileActionPlan.Target target : targets) {
            appendLine(markdown, row(rank, target));
            rank++;
        }
        appendLine(markdown, "");
    }

    private static void appendActions(StringBuilder markdown, List<String> actions) {
        if (actions == null || actions.isEmpty()) {
            return;
        }
        appendLine(markdown, "## Next Actions");
        appendLine(markdown, "");
        int index = 1;
        for (String action : actions) {
            appendLine(markdown, index + ". " + escapeListItem(action));
            index++;
        }
    }

    private static String row(
            int rank,
            TrainingReportRuntimeProfileActionPlan.Target target) {
        return "| " + rank
                + " | `" + target.kind().name() + "`"
                + " | `" + escapeTable(target.name()) + "`"
                + " | `" + target.priority().name() + "`"
                + " | `" + target.category().name() + "`"
                + " | " + target.totalMillis().stream()
                        .mapToObj(TrainingReportRuntimeProfileActionPlanMarkdown::formatMillis)
                        .findFirst()
                        .orElse("")
                + " | " + target.percentTotal().stream()
                        .mapToObj(TrainingReportRuntimeProfileActionPlanMarkdown::formatMillis)
                        .findFirst()
                        .orElse("")
                + " | `" + escapeTable(target.diagnosticCode()) + "`"
                + " |";
    }

    private static String formatMillis(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
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
