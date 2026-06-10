package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainingReportQualityProfileCiGateManifestMarkdown {
    private TrainingReportQualityProfileCiGateManifestMarkdown() {
    }

    static String render(Map<String, Object> manifest) {
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Quality Profile CI Gate Manifest");
        appendLine(markdown, "");
        appendLine(markdown, "**Profile:** `" + escapeInline(stringValue(manifest.get("profileId"), "unknown")) + "`");
        appendLine(markdown, "**Format:** `" + escapeInline(stringValue(manifest.get("format"), "unknown")) + "`");
        appendLine(markdown, "**Gate:** `" + status(manifest.get("passed")) + "`");
        appendLine(markdown, "**Validation:** `" + status(manifest.get("validationPassed")) + "`");
        appendLine(markdown, "**Promotion:** `" + status(manifest.get("promotionPassed")) + "`");
        appendLine(markdown, "**Artifacts verified:** `" + status(manifest.get("artifactsVerified")) + "`");
        appendLine(markdown, "**Artifact count:** `" + longValue(manifest.get("artifactCount"), 0L) + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Name | Kind | Report | File | Bytes | SHA-256 |");
        appendLine(markdown, "| --- | --- | --- | --- | ---: | --- |");
        for (Map<String, Object> artifact : mapList(manifest.get("artifacts"))) {
            appendLine(markdown, artifactRow(artifact));
        }
        appendLine(markdown, "");
        appendLine(markdown, "## Message");
        appendLine(markdown, "");
        appendLine(markdown, stringValue(manifest.get("message"), "No CI gate message recorded."));
        return markdown.toString();
    }

    private static String status(Object value) {
        return booleanValue(value) ? "PASS" : "FAIL";
    }

    private static String artifactRow(Map<String, Object> artifact) {
        return "| `" + escapeTable(stringValue(artifact.get("name"), "unknown")) + "`"
                + " | `" + escapeTable(stringValue(artifact.get("kind"), "unknown")) + "`"
                + " | `" + escapeTable(stringValue(artifact.get("reportName"), "")) + "`"
                + " | `" + escapeTable(stringValue(artifact.get("file"), "n/a")) + "`"
                + " | `" + escapeTable(stringValue(artifact.get("bytes"), "0")) + "`"
                + " | `" + escapeTable(shortSha(stringValue(artifact.get("sha256"), "n/a"))) + "` |";
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
                return Long.parseLong(text.trim());
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

    private static String escapeInline(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`");
    }

    private static String escapeTable(String value) {
        return escapeInline(value).replace("|", "\\|").replace("\n", " ");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
