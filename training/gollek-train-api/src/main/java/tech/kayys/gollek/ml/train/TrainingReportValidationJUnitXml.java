package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * JUnit XML renderer for single-report validation results.
 */
public final class TrainingReportValidationJUnitXml {
    private TrainingReportValidationJUnitXml() {
    }

    public static String render(TrainingReportValidationPolicy.Result result) {
        Objects.requireNonNull(result, "result must not be null");
        String markdown = result.markdown();
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.report.validation\" tests=\"1\" failures=\""
                + (result.passed() ? "0" : "1")
                + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "validation.passed", Boolean.toString(result.passed()));
        property(xml, "validation.failureCodes", String.join(",", result.failureCodes()));
        property(xml, "policy.maxDiagnosticSeverity", result.policy().maxDiagnosticSeverity().name());
        property(xml, "policy.requireRunHealthGate", Boolean.toString(result.policy().requireRunHealthGate()));
        property(xml, "policy.requireDataHealthGate", Boolean.toString(result.policy().requireDataHealthGate()));
        property(xml, "policy.requireDataHealthAvailable", Boolean.toString(result.policy().requireDataHealthAvailable()));
        property(xml, "policy.requireFreshDiagnostics", Boolean.toString(result.policy().requireFreshDiagnostics()));
        property(xml, "policy.requireValidation", Boolean.toString(result.policy().requireValidation()));
        property(xml, "policy.requireCheckpointIntegrity", Boolean.toString(result.policy().requireCheckpointIntegrity()));
        property(xml, "diagnostics.passed", Boolean.toString(result.diagnosticGate().passed()));
        property(xml, "diagnostics.highestSeverity", result.diagnosticGate().summary().highestSeverity());
        property(xml, "diagnostics.failingCodes", String.join(",", result.diagnosticGate().failingCodes()));
        property(xml, "runHealth.status", result.runHealth().status());
        property(xml, "runHealth.gatePassed", Boolean.toString(result.runHealth().gatePassed()));
        property(xml, "dataHealth.available", Boolean.toString(result.dataHealth().available()));
        property(xml, "dataHealth.gatePassed", Boolean.toString(result.dataHealth().gatePassed()));
        property(xml, "dataHealth.issueCount", Integer.toString(result.dataHealth().issueCount()));
        property(xml, "diagnostics.provenance", result.diagnosticProvenance().status());
        property(xml, "validation.available", Boolean.toString(result.validationAvailable()));
        property(xml, "checkpointIntegrity.available", Boolean.toString(result.checkpointIntegrityAvailable()));
        property(xml, "checkpointIntegrity.passed", Boolean.toString(result.checkpointIntegrityPassed()));
        property(xml, "checkpointIntegrity.failureKeys", String.join(",", result.checkpointIntegrityFailureKeys()));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.report\" name=\"validate training report\" time=\"0\">");
        if (result.failed()) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType(result))
                    + "\" message=\"" + escapeXml(result.message()) + "\">");
            appendLine(xml, escapeText(markdown));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
        appendLine(xml, "  </testcase>");
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static String failureType(TrainingReportValidationPolicy.Result result) {
        if (result.failureCodes().isEmpty()) {
            return "VALIDATION_FAILED";
        }
        return result.failureCodes().get(0).toUpperCase().replace('.', '_').replace('-', '_');
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static String escapeXml(String value) {
        return escapeText(value).replace("\"", "&quot;").replace("'", "&apos;");
    }

    private static String escapeText(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            switch (ch) {
                case '&' -> escaped.append("&amp;");
                case '<' -> escaped.append("&lt;");
                case '>' -> escaped.append("&gt;");
                default -> {
                    if (isValidXmlChar(ch)) {
                        escaped.append(ch);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static boolean isValidXmlChar(char ch) {
        return ch == 0x9
                || ch == 0xA
                || ch == 0xD
                || (ch >= 0x20 && ch <= 0xD7FF)
                || (ch >= 0xE000 && ch <= 0xFFFD);
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
