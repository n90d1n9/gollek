package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class TrainingReportPromotionGateArtifactReports {
    private TrainingReportPromotionGateArtifactReports() {
    }

    static String renderMarkdown(Map<String, Object> result) {
        Objects.requireNonNull(result, "result must not be null");
        Map<String, Object> decision = mapValue(result, "decision");
        Map<String, Object> review = mapValue(result, "review");
        Map<String, Object> artifacts = mapValue(result, "artifacts");
        Map<String, Object> verification = mapValue(result, "verification");
        Map<String, Object> sourceVerification = mapValue(result, "sourceVerification");

        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Promotion Gate");
        appendLine(markdown, "");
        appendLine(markdown, "**Gate:** `" + (booleanValue(result.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Status:** `" + stringValue(decision.get("status"), "UNKNOWN") + "`");
        appendLine(markdown, "**Promotable:** `" + booleanValue(result.get("promotable")) + "`");
        appendLine(markdown, "**Baseline:** `" + escapeInline(stringValue(decision.get("baseline"), "unknown")) + "`");
        String candidate = stringValue(decision.get("candidate"), "");
        appendLine(markdown, "**Candidate:** `" + escapeInline(candidate.isBlank() ? "none" : candidate) + "`");
        appendLine(markdown, "**Artifact verification:** `"
                + (booleanValue(verification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Source report verification:** `"
                + (booleanValue(sourceVerification.get("passed")) ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 | Verified |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        appendLine(markdown, artifactRow(
                "JSON review",
                stringValue(artifacts.get("jsonFile"), "n/a"),
                stringValue(artifacts.get("jsonSha256"), "n/a"),
                booleanValue(verification.get("jsonSha256Matches"))));
        appendLine(markdown, artifactRow(
                "Markdown review",
                stringValue(artifacts.get("markdownFile"), "n/a"),
                stringValue(artifacts.get("markdownSha256"), "n/a"),
                booleanValue(verification.get("markdownSha256Matches"))));

        List<Map<String, Object>> sourceReports = mapList(sourceVerification.get("reports"));
        if (!sourceReports.isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Source Reports");
            appendLine(markdown, "");
            appendLine(markdown, "| Role | Name | Path | Bytes | SHA-256 |");
            appendLine(markdown, "| --- | --- | --- | ---: | --- |");
            for (Map<String, Object> report : sourceReports) {
                appendLine(markdown, sourceReportRow(report));
            }
        }

        appendLine(markdown, "");
        appendLine(markdown, "## Review");
        appendLine(markdown, "");
        appendLine(markdown, "Baseline: `" + escapeInline(stringValue(review.get("baseline"), "unknown")) + "`");
        appendLine(markdown, "Candidates audited: `" + longValue(review.get("candidateCount"), 0L) + "`");
        appendLine(markdown, "Promotable candidates: `" + longValue(review.get("promotableCount"), 0L) + "`");
        appendLine(markdown, "Held candidates: `" + longValue(review.get("heldCount"), 0L) + "`");
        appendLine(markdown, "");
        appendLine(markdown, stringValue(result.get("message"), "No promotion gate message recorded."));

        appendListSection(markdown, "Reasons", stringList(decision.get("reasons")));
        appendListSection(markdown, "Verification Failures", stringList(verification.get("failures")));
        appendListSection(markdown, "Source Verification Failures", stringList(sourceVerification.get("failures")));
        return markdown.toString();
    }

    static String renderJunitXml(Map<String, Object> result) {
        Objects.requireNonNull(result, "result must not be null");
        Map<String, Object> decision = mapValue(result, "decision");
        Map<String, Object> artifacts = mapValue(result, "artifacts");
        Map<String, Object> verification = mapValue(result, "verification");
        Map<String, Object> sourceVerification = mapValue(result, "sourceVerification");
        boolean passed = booleanValue(result.get("passed"));
        String markdown = renderMarkdown(result);
        String baseline = stringValue(decision.get("baseline"), "unknown");
        String candidate = stringValue(decision.get("candidate"), "<none>");

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.promotion\" tests=\"1\" failures=\""
                + (passed ? "0" : "1")
                + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "gate.passed", Boolean.toString(passed));
        property(xml, "promotion.status", stringValue(decision.get("status"), "UNKNOWN"));
        property(xml, "promotion.promotable", Boolean.toString(booleanValue(decision.get("promotable"))));
        property(xml, "promotion.baseline", baseline);
        if (!candidate.isBlank() && !"<none>".equals(candidate)) {
            property(xml, "promotion.candidate", candidate);
        }
        property(xml, "artifacts.json", stringValue(artifacts.get("jsonFile"), ""));
        property(xml, "artifacts.markdown", stringValue(artifacts.get("markdownFile"), ""));
        property(xml, "artifacts.jsonSha256", stringValue(artifacts.get("jsonSha256"), ""));
        property(xml, "artifacts.markdownSha256", stringValue(artifacts.get("markdownSha256"), ""));
        property(xml, "artifacts.verified", Boolean.toString(booleanValue(verification.get("passed"))));
        property(xml, "sourceReports.verified", Boolean.toString(booleanValue(sourceVerification.get("passed"))));
        property(xml, "sourceReports.count", Long.toString(mapList(sourceVerification.get("reports")).size()));
        property(xml, "sourceReports.failures", Long.toString(stringList(sourceVerification.get("failures")).size()));
        appendLine(xml, "  </properties>");
        appendLine(xml, "  <testcase classname=\"gollek.training.promotion\" name=\""
                + escapeXml(baseline + " -> " + candidate)
                + "\" time=\"0\">");
        if (!passed) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType(decision, verification, sourceVerification))
                    + "\" message=\"" + escapeXml(stringValue(result.get("message"), "Promotion gate failed"))
                    + "\">");
            appendLine(xml, escapeText(markdown));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
        appendLine(xml, "  </testcase>");
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static String artifactRow(
            String label,
            String path,
            String sha256,
            boolean verified) {
        return "| " + escapeTable(label)
                + " | `" + escapeTable(path) + "`"
                + " | `" + escapeTable(shortSha(sha256)) + "`"
                + " | `" + (verified ? "yes" : "no") + "` |";
    }

    private static String sourceReportRow(Map<String, Object> report) {
        return "| " + escapeTable(stringValue(report.get("role"), "unknown"))
                + " | `" + escapeTable(stringValue(report.get("name"), "unknown")) + "`"
                + " | `" + escapeTable(stringValue(report.get("source"), "n/a")) + "`"
                + " | `" + escapeTable(stringValue(report.get("bytes"), "n/a")) + "`"
                + " | `" + escapeTable(shortSha(stringValue(report.get("sha256"), "n/a"))) + "` |";
    }

    private static void appendListSection(StringBuilder markdown, String title, List<String> values) {
        if (values.isEmpty()) {
            return;
        }
        appendLine(markdown, "");
        appendLine(markdown, "## " + title);
        appendLine(markdown, "");
        for (String value : values) {
            appendLine(markdown, "- " + escapeListItem(value));
        }
    }

    private static String failureType(
            Map<String, Object> decision,
            Map<String, Object> verification,
            Map<String, Object> sourceVerification) {
        if (!booleanValue(verification.get("passed"))) {
            return "ARTIFACT_VERIFICATION";
        }
        if (!booleanValue(sourceVerification.get("passed"))) {
            return "SOURCE_REPORT_VERIFICATION";
        }
        return stringValue(decision.get("status"), "PROMOTION_GATE");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Object snapshot = TrainingReportSnapshots.immutableSnapshot(map);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static List<Map<String, Object>> mapList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<Map<String, Object>> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item instanceof Map<?, ?> map) {
                values.add(immutableStringKeyMap(map));
            }
        }
        return List.copyOf(values);
    }

    private static List<String> stringList(Object value) {
        if (!(value instanceof Iterable<?> iterable)) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (Object item : iterable) {
            if (item != null) {
                String text = String.valueOf(item).trim();
                if (!text.isEmpty()) {
                    values.add(text);
                }
            }
        }
        return List.copyOf(values);
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), TrainingReportSnapshots.immutableSnapshot(entry.getValue()));
        }
        return Map.copyOf(values);
    }

    private static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool.booleanValue();
        }
        if (value instanceof String text) {
            return Boolean.parseBoolean(text);
        }
        return false;
    }

    private static long longValue(Object value, long fallback) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        if (value instanceof String text) {
            try {
                return Long.parseLong(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? fallback : text;
    }

    private static String shortSha(String sha256) {
        if (sha256 == null || sha256.isBlank()) {
            return "n/a";
        }
        String normalized = sha256.trim();
        return normalized.length() <= 12 ? normalized : normalized.substring(0, 12);
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
