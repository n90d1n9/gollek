package tech.kayys.gollek.ml.train;

/**
 * JUnit XML renderer for trainer performance gate CI integrations.
 */
final class TrainingReportPerformanceGateJUnitXml {
    private TrainingReportPerformanceGateJUnitXml() {
    }

    static String render(TrainingReportPerformanceGate.Result result) {
        if (result == null) {
            return "";
        }
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek-trainer-performance-gate\" tests=\"1\" failures=\""
                + (result.passed() ? "0" : "1") + "\">");
        appendLine(xml, "  <testcase classname=\"gollek.training\" name=\"performance-gate\">");
        if (!result.passed()) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType(result)) + "\" message=\""
                    + escapeXml(result.message()) + "\">" + escapeXml(findingCodes(result)) + "</failure>");
        }
        appendLine(xml, "  </testcase>");
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static String failureType(TrainingReportPerformanceGate.Result result) {
        return result.findings().stream()
                .map(TrainingReportPerformanceGate.Finding::severity)
                .anyMatch("failure"::equals)
                        ? "performance-failure"
                        : "performance-warning";
    }

    private static String findingCodes(TrainingReportPerformanceGate.Result result) {
        return String.join(
                ",",
                result.findings().stream()
                        .map(TrainingReportPerformanceGate.Finding::code)
                        .toList());
    }

    private static String escapeXml(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;")
                .replace("'", "&apos;");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
