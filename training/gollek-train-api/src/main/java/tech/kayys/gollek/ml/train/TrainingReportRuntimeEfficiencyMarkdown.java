package tech.kayys.gollek.ml.train;

import java.util.OptionalDouble;

/**
 * Renders compact trainer runtime-efficiency summaries for reports and notebooks.
 */
public final class TrainingReportRuntimeEfficiencyMarkdown {
    private TrainingReportRuntimeEfficiencyMarkdown() {
    }

    public static String render(TrainingReportRuntimeEfficiency efficiency) {
        if (efficiency == null || !efficiency.available()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "### Runtime Efficiency");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + efficiency.status().name() + "`");
        appendLine(markdown, "");
        appendLine(markdown, "| Signal | Value |");
        appendLine(markdown, "| --- | ---: |");
        appendLine(markdown, row("Measured trainer time", formatMillis(efficiency.measuredMillis())));
        appendLine(markdown, row("Wall-clock time", formatMillis(efficiency.wallMillis())));
        appendLine(markdown, row("Accounted wall time", formatPercent(efficiency.accountedPercent())));
        appendLine(markdown, row(
                "Largest wall overhead",
                "`" + escapeTable(efficiency.overheadScope()) + "` "
                        + formatMillis(efficiency.overheadMillis())
                        + " / "
                        + formatPercent(efficiency.overheadPercent())));
        appendLine(markdown, row(
                "Dominant bottleneck",
                "`" + escapeTable(efficiency.bottleneck()) + "` "
                        + formatPercent(efficiency.bottleneckPercent())));
        appendLine(markdown, row("Primary hotspot", "`" + escapeTable(efficiency.primaryHotspot()) + "`"));
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String row(String label, String value) {
        return "| " + escapeTable(label) + " | " + value + " |";
    }

    private static void appendLine(StringBuilder markdown, String line) {
        markdown.append(line).append('\n');
    }

    private static String formatMillis(OptionalDouble value) {
        return value.stream()
                .mapToObj(number -> String.format(java.util.Locale.ROOT, "`%.3f ms`", number))
                .findFirst()
                .orElse("n/a");
    }

    private static String formatPercent(OptionalDouble value) {
        return value.stream()
                .mapToObj(number -> String.format(java.util.Locale.ROOT, "`%.3f%%`", number))
                .findFirst()
                .orElse("n/a");
    }

    private static String escapeTable(String value) {
        return value == null ? "" : value.replace("|", "\\|");
    }
}
