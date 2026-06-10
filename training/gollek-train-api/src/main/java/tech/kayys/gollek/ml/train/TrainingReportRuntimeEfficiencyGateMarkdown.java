package tech.kayys.gollek.ml.train;

import java.util.Map;

/**
 * Markdown renderer for runtime-efficiency gate results.
 */
final class TrainingReportRuntimeEfficiencyGateMarkdown {
    private TrainingReportRuntimeEfficiencyGateMarkdown() {
    }

    static String render(TrainingReportRuntimeEfficiencyGate.Result result) {
        if (result == null) {
            return "# Runtime Efficiency Gate\n\n- Available: `false`\n- Passed: `true`\n";
        }
        StringBuilder markdown = new StringBuilder();
        markdown.append("# Runtime Efficiency Gate\n\n");
        markdown.append("- Available: `").append(result.available()).append("`\n");
        markdown.append("- Passed: `").append(result.passed()).append("`\n");
        markdown.append("- Findings: `").append(result.findings().size()).append("`\n");
        markdown.append("- Message: ").append(result.message()).append("\n\n");
        appendPolicy(markdown, result.policy());
        appendEfficiency(markdown, result.efficiency());
        appendFindings(markdown, result);
        return markdown.toString();
    }

    private static void appendPolicy(
            StringBuilder markdown,
            TrainingReportRuntimeEfficiencyGate.Policy policy) {
        markdown.append("## Policy\n\n");
        markdown.append("| Threshold | Value |\n");
        markdown.append("| --- | ---: |\n");
        markdown.append("| Minimum accounted wall time | `")
                .append(format(policy.minAccountedWallPercent()))
                .append("%` |\n");
        markdown.append("| Maximum wall-clock overhead | `")
                .append(format(policy.maxWallClockOverheadPercent()))
                .append("%` |\n");
        markdown.append("| Maximum bottleneck concentration | `")
                .append(format(policy.maxBottleneckPercent()))
                .append("%` |\n\n");
    }

    private static void appendEfficiency(StringBuilder markdown, Map<String, Object> efficiency) {
        markdown.append("## Efficiency\n\n");
        if (efficiency == null || !Boolean.TRUE.equals(efficiency.get("available"))) {
            markdown.append("Runtime efficiency metadata is not available.\n\n");
            return;
        }
        markdown.append("| Signal | Value |\n");
        markdown.append("| --- | ---: |\n");
        appendValue(markdown, "Status", efficiency.get("status"));
        appendMillis(markdown, "Measured trainer time", efficiency.get("measuredMillis"));
        appendMillis(markdown, "Wall-clock time", efficiency.get("wallMillis"));
        appendPercent(markdown, "Accounted wall time", efficiency.get("accountedPercent"));
        appendOverhead(markdown, efficiency);
        appendBottleneck(markdown, efficiency);
        appendValue(markdown, "Primary hotspot", efficiency.get("primaryHotspot"));
        markdown.append("\n");
    }

    private static void appendFindings(
            StringBuilder markdown,
            TrainingReportRuntimeEfficiencyGate.Result result) {
        markdown.append("## Findings\n\n");
        if (result.findings().isEmpty()) {
            markdown.append("No runtime efficiency budget warnings.\n");
            return;
        }
        markdown.append("| Severity | Code | Message | Action |\n");
        markdown.append("| --- | --- | --- | --- |\n");
        for (TrainingReportRuntimeEfficiencyGate.Finding finding : result.findings()) {
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

    private static void appendOverhead(StringBuilder markdown, Map<String, Object> efficiency) {
        Object scope = efficiency.get("overheadScope");
        Object millis = efficiency.get("overheadMillis");
        Object percent = efficiency.get("overheadPercent");
        markdown.append("| Largest wall overhead | `")
                .append(escapeCell(String.valueOf(scope == null ? "none" : scope)))
                .append("`");
        if (millis instanceof Number ms) {
            markdown.append(" `").append(format(ms.doubleValue())).append(" ms`");
        }
        if (percent instanceof Number pct) {
            markdown.append(" / `").append(format(pct.doubleValue())).append("%`");
        }
        markdown.append(" |\n");
    }

    private static void appendBottleneck(StringBuilder markdown, Map<String, Object> efficiency) {
        Object bottleneck = efficiency.get("bottleneck");
        Object percent = efficiency.get("bottleneckPercent");
        markdown.append("| Dominant bottleneck | `")
                .append(escapeCell(String.valueOf(bottleneck == null ? "none" : bottleneck)))
                .append("`");
        if (percent instanceof Number pct) {
            markdown.append(" `").append(format(pct.doubleValue())).append("%`");
        }
        markdown.append(" |\n");
    }

    private static void appendValue(StringBuilder markdown, String label, Object value) {
        if (value == null) {
            return;
        }
        markdown.append("| ").append(label).append(" | `").append(escapeCell(String.valueOf(value))).append("` |\n");
    }

    private static void appendMillis(StringBuilder markdown, String label, Object value) {
        if (value instanceof Number number) {
            markdown.append("| ").append(label).append(" | `").append(format(number.doubleValue())).append(" ms` |\n");
        }
    }

    private static void appendPercent(StringBuilder markdown, String label, Object value) {
        if (value instanceof Number number) {
            markdown.append("| ").append(label).append(" | `").append(format(number.doubleValue())).append("%` |\n");
        }
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.ROOT, "%.3f", value) : "n/a";
    }

    private static String escapeCell(String value) {
        return value == null ? "" : value.replace("|", "\\|").replace("\n", " ");
    }
}
