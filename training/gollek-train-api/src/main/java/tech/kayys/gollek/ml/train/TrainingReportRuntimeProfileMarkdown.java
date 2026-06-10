package tech.kayys.gollek.ml.train;

/**
 * Compact Markdown renderer for trainer runtime profile hotspots.
 */
public final class TrainingReportRuntimeProfileMarkdown {
    private TrainingReportRuntimeProfileMarkdown() {
    }

    public static boolean visible(TrainingReportRuntimeProfile profile) {
        return profile != null && profile.available();
    }

    public static String render(TrainingReportRuntimeProfile profile) {
        return render(profile, TrainingReportRuntimeInputProfile.empty());
    }

    static String render(
            TrainingReportRuntimeProfile profile,
            TrainingReportRuntimeInputProfile inputProfile) {
        if (!visible(profile)) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Runtime Profile");
        appendLine(markdown, "");
        profile.primaryGroup().ifPresent(group -> appendLine(
                markdown,
                "**Primary group:** `" + escapeInline(group.name()) + "`"
                        + group.totalMillis()
                                .stream()
                                .mapToObj(total -> " (" + formatMillis(total) + " ms total)")
                                .findFirst()
                                .orElse("")));
        if (profile.primaryGroup().isPresent()) {
            appendLine(markdown, "");
        }
        appendBalance(markdown, profile.balance());
        appendInputPipeline(markdown, inputProfile);
        appendGroups(markdown, profile);
        profile.primaryHotspot().ifPresent(hotspot -> appendLine(
                markdown,
                "**Primary hotspot:** `" + escapeInline(hotspot.phase()) + "`"
                        + hotspot.totalMillis()
                                .stream()
                                .mapToObj(total -> " (" + formatMillis(total) + " ms total)")
                                .findFirst()
                                .orElse("")));
        if (profile.primaryHotspot().isPresent()) {
            appendLine(markdown, "");
        }
        appendLine(markdown, "| Rank | Phase | Count | Total ms | Total % | Avg ms | Min ms | Max ms | Last ms | Stddev ms |");
        appendLine(markdown, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        int rank = 1;
        for (TrainingReportRuntimeProfile.Hotspot hotspot : profile.hotspots()) {
            appendLine(markdown, row(rank, hotspot));
            rank++;
        }
        appendLine(markdown, "");
        return markdown.toString();
    }

    static String renderInputPipeline(TrainingReportRuntimeInputProfile inputProfile) {
        if (inputProfile == null || !inputProfile.available()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendInputPipeline(markdown, inputProfile);
        return markdown.toString();
    }

    static String renderBalance(TrainingReportRuntimeProfile.Balance balance) {
        if (balance == null || !balance.available()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendBalance(markdown, balance);
        return markdown.toString();
    }

    private static void appendBalance(
            StringBuilder markdown,
            TrainingReportRuntimeProfile.Balance balance) {
        if (balance == null || !balance.available()) {
            return;
        }
        appendLine(markdown, "### Runtime Balance");
        appendLine(markdown, "");
        appendLine(markdown, "**Bottleneck group:** `" + escapeInline(balance.bottleneckGroup()) + "`"
                + balance.bottleneck().totalMillis()
                        .stream()
                        .mapToObj(total -> " (" + formatMillis(total) + " ms total)")
                        .findFirst()
                        .orElse(""));
        appendLine(markdown, "");
        appendLine(markdown, "| Bucket | Total ms | Total % |");
        appendLine(markdown, "| --- | ---: | ---: |");
        appendLine(markdown, balanceRow("input", balance.input()));
        appendLine(markdown, balanceRow("compute", balance.compute()));
        appendLine(markdown, balanceRow("train", balance.train()));
        appendLine(markdown, balanceRow("validation", balance.validation()));
        appendLine(markdown, balanceRow("optimizer", balance.optimizer()));
        appendLine(markdown, "");
    }

    private static String balanceRow(
            String bucket,
            TrainingReportRuntimeProfile.Bucket value) {
        return "| `" + escapeTable(bucket) + "`"
                + " | " + value.totalMillis()
                        .stream()
                        .mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis)
                        .findFirst()
                        .orElse("")
                + " | " + value.percentTotal()
                        .stream()
                        .mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis)
                        .findFirst()
                        .orElse("")
                + " |";
    }

    private static void appendInputPipeline(
            StringBuilder markdown,
            TrainingReportRuntimeInputProfile inputProfile) {
        if (inputProfile == null || !inputProfile.available()) {
            return;
        }
        appendLine(markdown, "### Input Pipeline");
        appendLine(markdown, "");
        inputProfile.bottleneckSummary().ifPresent(summary -> {
            appendLine(markdown, "**Input bottleneck:** `" + escapeInline(summary.scope()) + "."
                    + escapeInline(summary.stage()) + "()`"
                    + " (" + formatMillis(summary.stageTotalMillis()) + " ms of "
                    + formatMillis(summary.scopeTotalMillis()) + " ms "
                    + escapeInline(summary.scope()) + " input"
                    + summary.stagePercent()
                            .stream()
                            .mapToObj(percent -> ", " + formatMillis(percent) + "% of that scope")
                            .findFirst()
                            .orElse("")
                    + ")");
            appendLine(markdown, "");
        });
        appendLine(markdown, "| Scope | Iterator count | Iterator ms | HasNext count | HasNext ms | Next count | Next ms | Total ms |");
        appendLine(markdown, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        appendLine(markdown, inputRow(
                "train",
                inputProfile.trainIterator(),
                inputProfile.trainHasNext(),
                inputProfile.trainNext()));
        appendLine(markdown, inputRow(
                "validation",
                inputProfile.validationIterator(),
                inputProfile.validationHasNext(),
                inputProfile.validationNext()));
        appendLine(markdown, "");
    }

    private static String inputRow(
            String scope,
            TrainingReportRuntimeInputProfile.Stage iterator,
            TrainingReportRuntimeInputProfile.Stage hasNext,
            TrainingReportRuntimeInputProfile.Stage next) {
        return "| `" + escapeTable(scope) + "`"
                + " | " + formatCount(iterator)
                + " | " + formatTotalMillis(iterator)
                + " | " + formatCount(hasNext)
                + " | " + formatTotalMillis(hasNext)
                + " | " + formatCount(next)
                + " | " + formatTotalMillis(next)
                + " | " + formatStageTotal(iterator, hasNext, next)
                + " |";
    }

    private static void appendGroups(StringBuilder markdown, TrainingReportRuntimeProfile profile) {
        if (profile.groups().isEmpty()) {
            return;
        }
        appendLine(markdown, "| Group | Count | Total ms | Total % | Avg ms | Min ms | Max ms | Last ms | Stddev ms |");
        appendLine(markdown, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for (TrainingReportRuntimeProfile.Group group : profile.groups()) {
            appendLine(markdown, groupRow(group));
        }
        appendLine(markdown, "");
    }

    private static String groupRow(TrainingReportRuntimeProfile.Group group) {
        return "| `" + escapeTable(group.name()) + "`"
                + " | " + group.count().stream().mapToObj(String::valueOf).findFirst().orElse("")
                + " | " + group.totalMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.percentTotal().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.averageMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.minMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.maxMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.lastMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + group.stddevMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " |";
    }

    private static String row(int rank, TrainingReportRuntimeProfile.Hotspot hotspot) {
        return "| " + rank
                + " | `" + escapeTable(hotspot.phase()) + "`"
                + " | " + hotspot.count().stream().mapToObj(String::valueOf).findFirst().orElse("")
                + " | " + hotspot.totalMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.percentTotal().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.averageMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.minMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.maxMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.lastMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " | " + hotspot.stddevMillis().stream().mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis).findFirst().orElse("")
                + " |";
    }

    private static String formatMillis(double value) {
        return String.format(java.util.Locale.ROOT, "%.3f", value);
    }

    private static String formatCount(TrainingReportRuntimeInputProfile.Stage stage) {
        return stage.count().stream().mapToObj(String::valueOf).findFirst().orElse("");
    }

    private static String formatTotalMillis(TrainingReportRuntimeInputProfile.Stage stage) {
        return stage.totalMillis().stream()
                .mapToObj(TrainingReportRuntimeProfileMarkdown::formatMillis)
                .findFirst()
                .orElse("");
    }

    private static String formatStageTotal(TrainingReportRuntimeInputProfile.Stage first,
            TrainingReportRuntimeInputProfile.Stage second,
            TrainingReportRuntimeInputProfile.Stage third) {
        double total = 0.0;
        boolean present = false;
        for (TrainingReportRuntimeInputProfile.Stage stage : java.util.List.of(first, second, third)) {
            if (stage.totalMillis().isPresent()) {
                total += stage.totalMillis().orElseThrow();
                present = true;
            }
        }
        return present ? formatMillis(total) : "";
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
