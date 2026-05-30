package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * Compact Markdown renderer for single-report validation results.
 */
public final class TrainingReportValidationMarkdown {
    private TrainingReportValidationMarkdown() {
    }

    public static String render(TrainingReportValidationPolicy.Result result) {
        Objects.requireNonNull(result, "result must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Report Validation");
        appendLine(markdown, "");
        appendLine(markdown, "**Validation:** `" + (result.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Diagnostics:** `"
                + (result.diagnosticGate().passed() ? "PASS" : "FAIL") + "`"
                + " (highest `" + escapeInline(result.diagnosticGate().summary().highestSeverity())
                + "`, max allowed `" + result.policy().maxDiagnosticSeverity().name() + "`)");
        appendLine(markdown, "**Run health:** `"
                + escapeInline(result.runHealth().status()) + "`"
                + " (gate `" + (result.runHealth().gatePassed() ? "PASS" : "FAIL") + "`)");
        TrainingReportDataHealthSummary dataHealthSummary =
                TrainingReportDataHealthSummary.from(result.dataHealth());
        appendLine(markdown, "**Data health:** `"
                + escapeInline(dataHealthSummary.status()) + "`"
                + " (gate `" + (result.dataHealth().gatePassed() ? "PASS" : "FAIL") + "`)");
        appendLine(markdown, "**Diagnostics provenance:** `"
                + escapeInline(result.diagnosticProvenance().status()) + "`");
        appendLine(markdown, "**Validation data:** `"
                + (result.validationAvailable() ? "available" : "missing") + "`");
        appendLine(markdown, "**Checkpoint integrity:** `"
                + checkpointLabel(result) + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Policy");
        appendLine(markdown, "");
        appendLine(markdown, "| Setting | Value |");
        appendLine(markdown, "| --- | --- |");
        appendLine(markdown, policyRow("maxDiagnosticSeverity", result.policy().maxDiagnosticSeverity().name()));
        appendLine(markdown, policyRow("requireRunHealthGate", result.policy().requireRunHealthGate()));
        appendLine(markdown, policyRow("requireDataHealthGate", result.policy().requireDataHealthGate()));
        appendLine(markdown, policyRow("requireDataHealthAvailable", result.policy().requireDataHealthAvailable()));
        appendLine(markdown, policyRow("requireFreshDiagnostics", result.policy().requireFreshDiagnostics()));
        appendLine(markdown, policyRow("requireValidation", result.policy().requireValidation()));
        appendLine(markdown, policyRow("requireCheckpointIntegrity", result.policy().requireCheckpointIntegrity()));
        if (!result.failureCodes().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Failures");
            appendLine(markdown, "");
            for (String code : result.failureCodes()) {
                appendLine(markdown, "- `" + escapeInline(code) + "`");
            }
        }
        if (!result.reasons().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Reasons");
            appendLine(markdown, "");
            for (String reason : result.reasons()) {
                appendLine(markdown, "- " + escapeListItem(reason));
            }
        }
        if (!result.diagnosticGate().failingCodes().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Failing Diagnostics");
            appendLine(markdown, "");
            for (String code : result.diagnosticGate().failingCodes()) {
                appendLine(markdown, "- `" + escapeInline(code) + "`");
            }
        }
        appendLine(markdown, "");
        appendLine(markdown, result.message());
        return markdown.toString();
    }

    private static String checkpointLabel(TrainingReportValidationPolicy.Result result) {
        if (!result.checkpointIntegrityAvailable()) {
            return "not-recorded";
        }
        return result.checkpointIntegrityPassed() ? "PASS" : "FAIL";
    }

    private static String policyRow(String setting, boolean value) {
        return policyRow(setting, Boolean.toString(value));
    }

    private static String policyRow(String setting, String value) {
        return "| `" + escapeTable(setting) + "` | `" + escapeTable(value) + "` |";
    }

    private static String escapeListItem(String value) {
        return escapeInline(value).replace("\n", " ");
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
