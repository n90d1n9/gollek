package tech.kayys.gollek.ml.train;

/**
 * Markdown renderer for quality-profile catalog advisory recommendations.
 */
final class TrainingReportQualityProfileCatalogAdvisorMarkdown {
    private TrainingReportQualityProfileCatalogAdvisorMarkdown() {
    }

    static String render(TrainingReportQualityProfileCatalogAdvisor.Result result) {
        if (result == null) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Quality Profile Catalog Advice");
        appendLine(markdown, "");
        appendLine(markdown, "**Ready for CI:** `" + (result.readyForCi() ? "yes" : "no") + "`");
        appendLine(markdown, "**Catalog validation:** `" + (result.validation().passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Recommendations:** `" + result.recommendations().size() + "`");
        appendLine(markdown, "");
        if (result.recommendations().isEmpty()) {
            appendLine(markdown, "No catalog advisory recommendations.");
            return markdown.toString();
        }
        appendLine(markdown, "| Priority | Category | Code | Title |");
        appendLine(markdown, "|---|---|---|---|");
        for (TrainingReportRecommendation recommendation : result.recommendations()) {
            appendLine(markdown, "| `" + escape(recommendation.priority().name()) + "` | `"
                    + escape(recommendation.category().name()) + "` | `"
                    + escape(recommendation.diagnosticCode()) + "` | "
                    + escape(recommendation.title()) + " |");
        }
        appendLine(markdown, "");
        for (TrainingReportRecommendation recommendation : result.recommendations()) {
            appendLine(markdown, "## " + escape(recommendation.title()));
            appendLine(markdown, "");
            appendLine(markdown, "**Code:** `" + escape(recommendation.diagnosticCode()) + "`");
            appendLine(markdown, "**Priority:** `" + escape(recommendation.priority().name()) + "`");
            appendLine(markdown, "");
            appendLine(markdown, escape(recommendation.rationale()));
            appendLine(markdown, "");
            for (String action : recommendation.actions()) {
                appendLine(markdown, "- " + escape(action));
            }
            appendLine(markdown, "");
        }
        return markdown.toString();
    }

    private static void appendLine(StringBuilder markdown, String line) {
        markdown.append(line).append('\n');
    }

    private static String escape(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("|", "\\|").replace("\n", " ");
    }
}
