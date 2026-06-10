package tech.kayys.gollek.ml.train;

/**
 * Compact Markdown renderer for baseline-vs-candidate runtime-profile deltas.
 */
public final class TrainingReportRuntimeRegressionSummaryMarkdown {
    private TrainingReportRuntimeRegressionSummaryMarkdown() {
    }

    public static boolean visible(TrainingReportRuntimeRegressionSummary summary) {
        return summary != null && summary.available();
    }

    public static String render(TrainingReportRuntimeRegressionSummary summary) {
        if (!visible(summary)) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Runtime Regression Summary");
        appendLine(markdown, "");
        appendLine(markdown, "**Regressed:** `" + (summary.regressed() ? "yes" : "no") + "`");
        appendLine(markdown, "");
        appendLine(markdown, "| Scope | Key | Baseline avg ms | Candidate avg ms | Ratio | Threshold | Regressed |");
        appendLine(markdown, "| --- | --- | ---: | ---: | ---: | ---: | --- |");
        summary.primaryGroupAverage().ifPresent(entry -> appendLine(markdown, row("primary group", entry)));
        summary.primaryHotspotAverage().ifPresent(entry -> appendLine(markdown, row("primary hotspot", entry)));
        appendEfficiencyRows(markdown, summary);
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static void appendEfficiencyRows(
            StringBuilder markdown,
            TrainingReportRuntimeRegressionSummary summary) {
        if (summary.accountedWallTime().isEmpty()
                && summary.wallClockOverhead().isEmpty()
                && summary.dominantBottleneck().isEmpty()) {
            return;
        }
        appendLine(markdown, "");
        appendLine(markdown, "| Efficiency Signal | Key | Baseline | Candidate | Delta | Threshold | Regressed |");
        appendLine(markdown, "| --- | --- | ---: | ---: | ---: | ---: | --- |");
        summary.accountedWallTime().ifPresent(entry -> appendLine(markdown, efficiencyRow("accounted wall time", entry)));
        summary.wallClockOverhead().ifPresent(entry -> appendLine(markdown, efficiencyRow("wall-clock overhead", entry)));
        summary.dominantBottleneck().ifPresent(entry -> appendLine(markdown, efficiencyRow("dominant bottleneck", entry)));
    }

    private static String row(String scope, TrainingReportRuntimeRegressionSummary.Entry entry) {
        return "| " + escapeTable(scope)
                + " | `" + escapeTable(entry.key()) + "`"
                + " | " + format(entry.baselineAverageMillis())
                + " | " + format(entry.candidateAverageMillis())
                + " | " + format(entry.ratio())
                + " | " + format(entry.threshold())
                + " | `" + (entry.regressed() ? "yes" : "no") + "`"
                + " |";
    }

    private static String efficiencyRow(
            String scope,
            TrainingReportRuntimeRegressionSummary.EfficiencyEntry entry) {
        return "| " + escapeTable(scope)
                + " | `" + escapeTable(entry.key()) + "`"
                + " | " + format(entry.baselineValue()) + unit(entry)
                + " | " + format(entry.candidateValue()) + unit(entry)
                + " | " + format(entry.delta()) + unit(entry)
                + " | " + format(entry.threshold()) + unit(entry)
                + " | `" + (entry.regressed() ? "yes" : "no") + "`"
                + " |";
    }

    private static String unit(TrainingReportRuntimeRegressionSummary.EfficiencyEntry entry) {
        return "percent".equals(entry.unit()) ? "%" : "";
    }

    private static String format(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
    }

    private static String escapeTable(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`").replace("|", "\\|").replace("\n", " ");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
