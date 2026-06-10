package tech.kayys.gollek.ml.train;

import java.util.Map;

final class TrainingReportRuntimeProfileBudgetGateMarkdown {
    private TrainingReportRuntimeProfileBudgetGateMarkdown() {
    }

    static String render(TrainingReportRuntimeProfileBudgetGate.Result result) {
        if (result == null) {
            return "# Runtime Profile Budget Gate\n\n- Available: `false`\n- Passed: `true`\n";
        }
        StringBuilder markdown = new StringBuilder();
        markdown.append("# Runtime Profile Budget Gate\n\n");
        markdown.append("- Available: `").append(result.available()).append("`\n");
        markdown.append("- Passed: `").append(result.passed()).append("`\n");
        markdown.append("- Findings: `").append(result.findings().size()).append("`\n");
        markdown.append("- Message: ").append(result.message()).append("\n\n");
        appendPolicy(markdown, result.policy());
        appendRuntimeSummary(markdown, result.runtimeProfile());
        appendFindings(markdown, result);
        return markdown.toString();
    }

    private static void appendPolicy(
            StringBuilder markdown,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) {
        markdown.append("## Policy\n\n");
        markdown.append("| Threshold | Value |\n");
        markdown.append("| --- | ---: |\n");
        markdown.append("| Primary group | `").append(format(policy.maxPrimaryGroupPercent())).append("%` |\n");
        markdown.append("| Primary hotspot | `").append(format(policy.maxPrimaryHotspotPercent())).append("%` |\n");
        markdown.append("| Primary hotspot total | `")
                .append(formatBudget(policy.maxPrimaryHotspotTotalMillis()))
                .append("` |\n");
        markdown.append("| Input balance | `").append(format(policy.maxInputBalancePercent())).append("%` |\n");
        markdown.append("| Optimizer balance | `").append(format(policy.maxOptimizerBalancePercent())).append("%` |\n");
        markdown.append("| Validation balance | `").append(format(policy.maxValidationBalancePercent())).append("%` |\n");
        markdown.append("| Wall-clock overhead | `")
                .append(format(policy.maxWallClockOverheadPercent()))
                .append("%` |\n");
        markdown.append("| Wall-clock overhead total | `")
                .append(formatBudget(policy.maxWallClockOverheadMillis()))
                .append("` |\n\n");
    }

    private static void appendRuntimeSummary(StringBuilder markdown, Map<String, Object> runtimeProfile) {
        if (runtimeProfile == null || !Boolean.TRUE.equals(runtimeProfile.get("available"))) {
            markdown.append("## Runtime Summary\n\nRuntime profile timings are not available.\n\n");
            return;
        }
        markdown.append("## Runtime Summary\n\n");
        markdown.append("| Metric | Value |\n");
        markdown.append("| --- | ---: |\n");
        appendPrimary(markdown, "Primary group", runtimeProfile.get("primaryGroup"), "name");
        appendPrimary(markdown, "Primary hotspot", runtimeProfile.get("primaryHotspot"), "phase");
        appendBalance(markdown, runtimeProfile.get("balance"));
        appendWallClock(markdown, runtimeProfile.get("wallClock"));
        markdown.append("\n");
    }

    private static void appendFindings(
            StringBuilder markdown,
            TrainingReportRuntimeProfileBudgetGate.Result result) {
        markdown.append("## Findings\n\n");
        if (result.findings().isEmpty()) {
            markdown.append("No runtime profile budget warnings.\n");
            return;
        }
        markdown.append("| Severity | Code | Message | Action |\n");
        markdown.append("| --- | --- | --- | --- |\n");
        for (TrainingReportRuntimeProfileBudgetGate.Finding finding : result.findings()) {
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

    private static void appendPrimary(
            StringBuilder markdown,
            String label,
            Object value,
            String nameKey) {
        if (!(value instanceof Map<?, ?> map)) {
            return;
        }
        Object name = map.get(nameKey);
        if (name == null || String.valueOf(name).isBlank()) {
            return;
        }
        markdown.append("| ").append(label).append(" | `").append(escapeCell(String.valueOf(name))).append("`");
        Object totalMillis = map.get("totalMillis");
        Object percentTotal = map.get("percentTotal");
        if (totalMillis instanceof Number total) {
            markdown.append(" (`").append(format(total.doubleValue())).append(" ms`");
            if (percentTotal instanceof Number percent) {
                markdown.append(", `").append(format(percent.doubleValue())).append("%`");
            }
            markdown.append(")");
        }
        markdown.append(" |\n");
    }

    private static void appendBalance(StringBuilder markdown, Object value) {
        if (!(value instanceof Map<?, ?> balance) || !Boolean.TRUE.equals(balance.get("available"))) {
            return;
        }
        Object bottleneck = balance.get("bottleneckGroup");
        if (bottleneck != null && !String.valueOf(bottleneck).isBlank()) {
            markdown.append("| Balance bottleneck | `").append(escapeCell(String.valueOf(bottleneck))).append("` |\n");
        }
        appendBalanceBucket(markdown, "Input balance", balance.get("input"));
        appendBalanceBucket(markdown, "Optimizer balance", balance.get("optimizer"));
        appendBalanceBucket(markdown, "Validation balance", balance.get("validation"));
    }

    private static void appendBalanceBucket(StringBuilder markdown, String label, Object value) {
        if (!(value instanceof Map<?, ?> bucket) || !Boolean.TRUE.equals(bucket.get("available"))) {
            return;
        }
        Object totalMillis = bucket.get("totalMillis");
        Object percentTotal = bucket.get("percentTotal");
        if (!(percentTotal instanceof Number percent)) {
            return;
        }
        markdown.append("| ").append(label).append(" | `").append(format(percent.doubleValue())).append("%`");
        if (totalMillis instanceof Number total) {
            markdown.append(" (`").append(format(total.doubleValue())).append(" ms`)");
        }
        markdown.append(" |\n");
    }

    private static void appendWallClock(StringBuilder markdown, Object value) {
        if (!(value instanceof Map<?, ?> wallClock) || !Boolean.TRUE.equals(wallClock.get("available"))) {
            return;
        }
        Object scope = wallClock.get("primaryOverheadScope");
        Object primary = wallClock.get("primaryOverhead");
        if (!(primary instanceof Map<?, ?> overhead) || scope == null || String.valueOf(scope).isBlank()) {
            return;
        }
        Object overheadMillis = overhead.get("overheadMillis");
        Object overheadPercent = overhead.get("overheadPercent");
        if (!(overheadMillis instanceof Number millis) && !(overheadPercent instanceof Number)) {
            return;
        }
        markdown.append("| Wall overhead | `").append(escapeCell(String.valueOf(scope))).append("`");
        if (overheadMillis instanceof Number millisValue) {
            markdown.append(" (`").append(format(millisValue.doubleValue())).append(" ms`");
            if (overheadPercent instanceof Number percentValue) {
                markdown.append(", `").append(format(percentValue.doubleValue())).append("%`");
            }
            markdown.append(")");
        }
        markdown.append(" |\n");
    }

    private static String formatBudget(double value) {
        return Double.isFinite(value) ? format(value) + " ms" : "unlimited";
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(java.util.Locale.ROOT, "%.3f", value) : "n/a";
    }

    private static String escapeCell(String value) {
        return value == null ? "" : value.replace("|", "\\|").replace("\n", " ");
    }
}
