package tech.kayys.gollek.ml.train;

import java.util.Objects;

/**
 * JUnit XML renderer for promotion gate results.
 */
public final class TrainingReportPromotionGateJUnitXml {
    private TrainingReportPromotionGateJUnitXml() {
    }

    public static String render(TrainingReportPromotionGate.Result result) {
        Objects.requireNonNull(result, "result must not be null");
        TrainingReportPortfolio.PromotionDecision decision = result.decision();
        TrainingReportPromotionArtifacts.ArtifactBundle artifacts = result.artifacts();
        TrainingReportPromotionArtifacts.ArtifactVerification verification = result.verification();
        TrainingReportPromotionArtifacts.SourceVerification sourceVerification = result.sourceVerification();
        String testName = testName(decision);
        String failureType = failureType(result, sourceVerification);

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\""
                + (result.passed() ? "0" : "1")
                + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "gate.passed", Boolean.toString(result.passed()));
        property(xml, "promotion.status", decision.status().name());
        property(xml, "promotion.promotable", Boolean.toString(decision.promotable()));
        property(xml, "promotion.baseline", decision.baseline().name());
        decision.candidate().ifPresent(candidate -> property(xml, "promotion.candidate", candidate.name()));
        property(xml, "artifacts.json", artifacts.jsonFile().toString());
        property(xml, "artifacts.markdown", artifacts.markdownFile().toString());
        property(xml, "artifacts.jsonSha256", artifacts.jsonSha256());
        property(xml, "artifacts.markdownSha256", artifacts.markdownSha256());
        property(xml, "artifacts.verified", Boolean.toString(verification.passed()));
        property(xml, "sourceReports.verified", Boolean.toString(sourceVerification.passed()));
        property(xml, "sourceReports.count", Integer.toString(sourceVerification.reports().size()));
        property(xml, "sourceReports.failures", Integer.toString(sourceVerification.failures().size()));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.promotion\" name=\""
                + escapeXml(testName)
                + "\" time=\"0\">");
        if (!result.passed()) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType)
                    + "\" message=\"" + escapeXml(result.message()) + "\">");
            appendLine(xml, escapeText(result.markdown()));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(result.markdown()) + "</system-out>");
        appendLine(xml, "  </testcase>");
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static String testName(TrainingReportPortfolio.PromotionDecision decision) {
        String candidate = decision.candidate()
                .map(TrainingReportPortfolio.Entry::name)
                .orElse("<none>");
        return decision.baseline().name() + " -> " + candidate;
    }

    private static String failureType(
            TrainingReportPromotionGate.Result result,
            TrainingReportPromotionArtifacts.SourceVerification sourceVerification) {
        if (!result.verification().passed()) {
            return "ARTIFACT_VERIFICATION";
        }
        if (!sourceVerification.passed()) {
            return "SOURCE_REPORT_VERIFICATION";
        }
        return result.decision().status().name();
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
