package tech.kayys.gollek.ml.train;

/**
 * Markdown renderer for runtime-regression gate results.
 */
final class TrainingReportRuntimeRegressionGateMarkdown {
    private TrainingReportRuntimeRegressionGateMarkdown() {
    }

    static String render(TrainingReportRuntimeRegressionGate.Result result) {
        if (result == null) {
            return "# Runtime Regression Gate\n\n- Available: `false`\n- Passed: `true`\n";
        }
        StringBuilder markdown = new StringBuilder();
        markdown.append("# Runtime Regression Gate\n\n");
        markdown.append("- Available: `").append(result.available()).append("`\n");
        markdown.append("- Passed: `").append(result.passed()).append("`\n");
        markdown.append("- Findings: `").append(result.findings().size()).append("`\n");
        markdown.append("- Message: ").append(result.message()).append("\n\n");
        appendSummary(markdown, result.runtimeRegression());
        appendFindings(markdown, result);
        return markdown.toString();
    }

    private static void appendSummary(
            StringBuilder markdown,
            TrainingReportRuntimeRegressionSummary summary) {
        String rendered = TrainingReportRuntimeRegressionSummaryMarkdown.render(summary);
        if (rendered.isBlank()) {
            markdown.append("## Runtime Regression Summary\n\n");
            markdown.append("Runtime regression metadata is not available.\n\n");
            return;
        }
        markdown.append(rendered).append("\n");
    }

    private static void appendFindings(
            StringBuilder markdown,
            TrainingReportRuntimeRegressionGate.Result result) {
        markdown.append("## Gate Findings\n\n");
        if (result.findings().isEmpty()) {
            markdown.append("No runtime regression findings.\n");
            return;
        }
        markdown.append("| Severity | Code | Message | Action |\n");
        markdown.append("| --- | --- | --- | --- |\n");
        for (TrainingReportRuntimeRegressionGate.Finding finding : result.findings()) {
            markdown.append("| `")
                    .append(escapeCell(finding.severity()))
                    .append("` | `")
                    .append(escapeCell(finding.code()))
                    .append("` | ")
                    .append(escapeCell(finding.message()))
                    .append(" | ")
                    .append(escapeCell(finding.action()))
                    .append(" |\n");
        }
    }

    private static String escapeCell(String value) {
        return value == null ? "" : value.replace("|", "\\|").replace("\n", " ");
    }
}
