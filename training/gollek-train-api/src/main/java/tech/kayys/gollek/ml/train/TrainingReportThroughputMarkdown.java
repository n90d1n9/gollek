package tech.kayys.gollek.ml.train;

/**
 * Compact Markdown renderer for trainer throughput counters and rates.
 */
public final class TrainingReportThroughputMarkdown {
    private TrainingReportThroughputMarkdown() {
    }

    public static String render(TrainingReportThroughput throughput) {
        if (throughput == null || !throughput.available()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Throughput");
        appendLine(markdown, "");
        appendLine(markdown, "| Phase | Batches | Samples | Compute ms | Samples/s | Batches/s | Avg batch ms |");
        appendLine(markdown, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
        appendLine(markdown, row(throughput.train()));
        appendLine(markdown, row(throughput.validation()));
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String row(TrainingReportThroughput.Phase phase) {
        return "| `" + escapeTable(phase.name()) + "`"
                + " | " + phase.batchCount().stream().mapToObj(String::valueOf).findFirst().orElse("")
                + " | " + phase.sampleCount().stream().mapToObj(String::valueOf).findFirst().orElse("")
                + " | " + format(phase.computeMillis())
                + " | " + format(phase.samplesPerSecond())
                + " | " + format(phase.batchesPerSecond())
                + " | " + format(phase.averageBatchMillis())
                + " |";
    }

    private static String format(java.util.OptionalDouble value) {
        return value.stream()
                .mapToObj(number -> String.format(java.util.Locale.ROOT, "%.3f", number))
                .findFirst()
                .orElse("");
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
