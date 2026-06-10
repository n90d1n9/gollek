package tech.kayys.gollek.ml.train;

/**
 * Markdown renderer for quality-profile catalog preflight validation results.
 */
final class TrainingReportQualityProfileCatalogValidatorMarkdown {
    private TrainingReportQualityProfileCatalogValidatorMarkdown() {
    }

    static String render(TrainingReportQualityProfileCatalogValidator.Result result) {
        TrainingReportQualityProfileCatalogValidator.Result resolved = result == null
                ? new TrainingReportQualityProfileCatalogValidator.Result(false, 0, java.util.List.of(), java.util.List.of())
                : result;
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Quality Profile Catalog Validation");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + (resolved.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Message:** " + escape(resolved.message()));
        appendLine(markdown, "**Profiles:** `" + resolved.profileCount() + "`");
        if (!resolved.profileIds().isEmpty()) {
            appendLine(markdown, "**Profile IDs:** `" + escape(String.join(", ", resolved.profileIds())) + "`");
        }
        appendLine(markdown, "");
        if (resolved.issues().isEmpty()) {
            appendLine(markdown, "No catalog validation issues.");
            return markdown.toString();
        }
        appendLine(markdown, "| Severity | Code | Path | Message | Action |");
        appendLine(markdown, "|---|---|---|---|---|");
        for (TrainingReportQualityProfileCatalogValidator.Issue issue : resolved.issues()) {
            appendLine(markdown, "| `" + escape(issue.severity()) + "` | `"
                    + escape(issue.code()) + "` | `" + escape(issue.path()) + "` | "
                    + escape(issue.message()) + " | " + escape(issue.action()) + " |");
        }
        return markdown.toString();
    }

    private static void appendLine(StringBuilder markdown, String line) {
        markdown.append(line).append('\n');
    }

    private static String escape(String value) {
        if (value == null || value.isEmpty()) {
            return "";
        }
        return value.replace("|", "\\|")
                .replace("\n", " ")
                .replace("\r", " ");
    }
}
