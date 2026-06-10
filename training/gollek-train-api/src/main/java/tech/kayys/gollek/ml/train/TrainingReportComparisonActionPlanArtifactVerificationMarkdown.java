package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * Markdown renderer for comparison action-plan artifact verification results.
 */
public final class TrainingReportComparisonActionPlanArtifactVerificationMarkdown {
    private TrainingReportComparisonActionPlanArtifactVerificationMarkdown() {
    }

    public static String render(TrainingReportComparisonActionPlanArtifacts.Verification verification) {
        TrainingReportComparisonActionPlanArtifacts.Verification resolved =
                Objects.requireNonNull(verification, "verification must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Comparison Action-Plan Artifact Verification");
        appendLine(markdown, "");
        appendLine(markdown, "**Status:** `" + (resolved.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Summary:** " + escapeText(resolved.summary()));
        appendLine(markdown, "**Manifest:** `" + escapeInlineCode(resolved.manifestFile().toString()) + "`");
        appendLine(markdown, "**Failures:** `" + resolved.failureCount() + "`");
        appendLine(markdown, "");
        if (resolved.hasFailures()) {
            appendLine(markdown, "### Failures");
            appendLine(markdown, "");
            for (String failure : resolved.failures()) {
                appendLine(markdown, "- " + escapeText(failure));
            }
            appendLine(markdown, "");
        }
        return markdown.toString();
    }

    private static String escapeText(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.replace("\n", " ").trim();
    }

    private static String escapeInlineCode(String value) {
        return escapeText(value).replace("`", "\\`");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
