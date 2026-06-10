package tech.kayys.gollek.ml.train;

import java.util.Map;

final class TrainingReportRuntimeInputProfileGateMarkdown {
    private TrainingReportRuntimeInputProfileGateMarkdown() {
    }

    static String render(TrainingReportRuntimeInputProfileGate.Result result) {
        if (result == null) {
            return "# Runtime Input Profile Gate\n\n- Available: `false`\n- Passed: `true`\n";
        }
        StringBuilder markdown = new StringBuilder();
        markdown.append("# Runtime Input Profile Gate\n\n");
        markdown.append("- Available: `").append(result.available()).append("`\n");
        markdown.append("- Passed: `").append(result.passed()).append("`\n");
        markdown.append("- Findings: `").append(result.findings().size()).append("`\n");
        markdown.append("- Message: ").append(result.message()).append("\n\n");
        appendPolicy(markdown, result.policy());
        appendInputSummary(markdown, result.inputProfile());
        appendFindings(markdown, result);
        return markdown.toString();
    }

    private static void appendPolicy(
            StringBuilder markdown,
            TrainingReportRuntimeInputProfileGate.Policy policy) {
        markdown.append("## Policy\n\n");
        markdown.append("| Threshold | Value |\n");
        markdown.append("| --- | ---: |\n");
        markdown.append("| Dominant scope | `").append(format(policy.maxDominantScopePercent())).append("%` |\n");
        markdown.append("| Dominant stage | `").append(format(policy.maxDominantStagePercent())).append("%` |\n");
        markdown.append("| Train/validation input ratio | `")
                .append(format(policy.maxTrainToValidationTotalRatio()))
                .append("x` |\n\n");
    }

    private static void appendInputSummary(StringBuilder markdown, Map<String, Object> inputProfile) {
        if (inputProfile == null || !Boolean.TRUE.equals(inputProfile.get("available"))) {
            markdown.append("## Input Summary\n\nRuntime input-profile timings are not available.\n\n");
            return;
        }
        markdown.append("## Input Summary\n\n");
        markdown.append("| Metric | Value |\n");
        markdown.append("| --- | ---: |\n");
        appendRow(markdown, "Total input", inputProfile.get("totalMillis"), " ms");
        appendRow(markdown, "Dominant scope", inputProfile.get("dominantScope"), "");
        appendRow(markdown, "Dominant scope total", inputProfile.get("dominantScopeTotalMillis"), " ms");
        appendRow(markdown, "Dominant scope percent", inputProfile.get("dominantScopePercent"), "%");
        appendRow(markdown, "Train/validation input ratio", inputProfile.get("trainToValidationTotalRatio"), "x");
        markdown.append("\n");
    }

    private static void appendFindings(
            StringBuilder markdown,
            TrainingReportRuntimeInputProfileGate.Result result) {
        markdown.append("## Findings\n\n");
        if (result.findings().isEmpty()) {
            markdown.append("No runtime input bottleneck warnings.\n");
            return;
        }
        markdown.append("| Severity | Code | Message | Action |\n");
        markdown.append("| --- | --- | --- | --- |\n");
        for (TrainingReportRuntimeInputProfileGate.Finding finding : result.findings()) {
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

    private static void appendRow(StringBuilder markdown, String label, Object value, String suffix) {
        if (value == null) {
            return;
        }
        markdown.append("| ")
                .append(label)
                .append(" | `")
                .append(formatValue(value))
                .append(suffix)
                .append("` |\n");
    }

    private static String formatValue(Object value) {
        if (value instanceof Number number) {
            return format(number.doubleValue());
        }
        return String.valueOf(value);
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.ROOT, "%.3f", value) : "n/a";
    }

    private static String escapeCell(String value) {
        return value == null ? "" : value.replace("|", "\\|").replace("\n", " ");
    }
}
