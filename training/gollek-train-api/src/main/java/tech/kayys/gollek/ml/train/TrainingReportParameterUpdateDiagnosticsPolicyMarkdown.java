package tech.kayys.gollek.ml.train;

/**
 * Compact Markdown renderer for parameter-update diagnostic sampling policy.
 */
public final class TrainingReportParameterUpdateDiagnosticsPolicyMarkdown {
    private TrainingReportParameterUpdateDiagnosticsPolicyMarkdown() {
    }

    public static boolean visible(TrainingReportParameterUpdateDiagnosticsPolicy policy) {
        return policy != null && (policy.enabled() || policy.sampled() || policy.intervalSteps() > 1);
    }

    public static String render(TrainingReportParameterUpdateDiagnosticsPolicy policy) {
        if (!visible(policy)) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Parameter Update Diagnostics");
        appendLine(markdown, "");
        appendLine(markdown, "| Enabled | Sampled | Interval Steps |");
        appendLine(markdown, "| --- | --- | ---: |");
        appendLine(markdown, "| `" + yesNo(policy.enabled()) + "` | `"
                + yesNo(policy.sampled()) + "` | " + policy.intervalSteps() + " |");
        appendLine(markdown, "");
        appendLine(markdown, summary(policy));
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String summary(TrainingReportParameterUpdateDiagnosticsPolicy policy) {
        if (policy.enabled() && policy.sampled()) {
            return "Exact parameter-update diagnostics are sampled every "
                    + policy.intervalSteps() + " optimizer step(s).";
        }
        if (policy.enabled()) {
            return "Exact parameter-update diagnostics run every optimizer step.";
        }
        return "Exact parameter-update diagnostics are disabled.";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
