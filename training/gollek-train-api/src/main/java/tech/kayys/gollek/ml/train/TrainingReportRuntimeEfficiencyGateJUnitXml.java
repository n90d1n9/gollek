package tech.kayys.gollek.ml.train;

import java.util.Objects;
import java.util.stream.Collectors;

/**
 * JUnit XML renderer for runtime-efficiency gate results.
 */
public final class TrainingReportRuntimeEfficiencyGateJUnitXml {
    private TrainingReportRuntimeEfficiencyGateJUnitXml() {
    }

    public static String render(TrainingReportRuntimeEfficiencyGate.Result result) {
        Objects.requireNonNull(result, "result must not be null");
        String markdown = result.markdown();
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.runtime.efficiency\" tests=\"1\" failures=\""
                + (result.passed() ? "0" : "1")
                + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "gate.available", Boolean.toString(result.available()));
        property(xml, "gate.passed", Boolean.toString(result.passed()));
        property(xml, "gate.findingCount", Integer.toString(result.findings().size()));
        property(xml, "gate.findingCodes", findingCodes(result));
        property(xml, "policy.minAccountedWallPercent",
                Double.toString(result.policy().minAccountedWallPercent()));
        property(xml, "policy.maxWallClockOverheadPercent",
                Double.toString(result.policy().maxWallClockOverheadPercent()));
        property(xml, "policy.maxBottleneckPercent",
                Double.toString(result.policy().maxBottleneckPercent()));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.runtime.efficiency\" "
                + "name=\"validate runtime efficiency\" time=\"0\">");
        if (!result.passed()) {
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

    private static String findingCodes(TrainingReportRuntimeEfficiencyGate.Result result) {
        return result.findings().stream()
                .map(TrainingReportRuntimeEfficiencyGate.Finding::code)
                .collect(Collectors.joining(","));
    }

    private static String failureType(TrainingReportRuntimeEfficiencyGate.Result result) {
        if (result.findings().isEmpty()) {
            return "RUNTIME_EFFICIENCY_GATE_FAILED";
        }
        return result.findings().getFirst().code().toUpperCase().replace('-', '_').replace('.', '_');
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
