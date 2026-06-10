package tech.kayys.gollek.ml.train;

/**
 * JUnit XML renderer for quality-profile catalog validation preflight results.
 */
final class TrainingReportQualityProfileCatalogValidatorJUnitXml {
    private TrainingReportQualityProfileCatalogValidatorJUnitXml() {
    }

    static String render(TrainingReportQualityProfileCatalogValidator.Result result) {
        if (result == null) {
            return "";
        }
        String markdown = result.markdown();
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.quality-profile-catalog.validation\" tests=\"1\" failures=\""
                + (result.passed() ? "0" : "1") + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "catalog.validJson", Boolean.toString(result.validJson()));
        property(xml, "catalog.passed", Boolean.toString(result.passed()));
        property(xml, "catalog.profileCount", Integer.toString(result.profileCount()));
        property(xml, "catalog.profileIds", String.join(",", result.profileIds()));
        property(xml, "catalog.errorCount", Integer.toString(result.errors().size()));
        property(xml, "catalog.warningCount", Integer.toString(result.warnings().size()));
        property(xml, "catalog.issueCodes", issueCodes(result));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.quality-profile-catalog\" "
                + "name=\"validate quality profile catalog\" time=\"0\">");
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

    private static String failureType(TrainingReportQualityProfileCatalogValidator.Result result) {
        return result.errors().stream()
                .map(TrainingReportQualityProfileCatalogValidator.Issue::code)
                .findFirst()
                .orElse("catalog.validation_failed")
                .toUpperCase()
                .replace('.', '_')
                .replace('-', '_');
    }

    private static String issueCodes(TrainingReportQualityProfileCatalogValidator.Result result) {
        return String.join(
                ",",
                result.issues().stream()
                        .map(TrainingReportQualityProfileCatalogValidator.Issue::code)
                        .toList());
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
