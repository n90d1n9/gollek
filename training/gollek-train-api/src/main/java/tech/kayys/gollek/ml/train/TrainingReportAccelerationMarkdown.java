package tech.kayys.gollek.ml.train;

/**
 * Compact Markdown renderer for trainer accelerator placement and kernel usage.
 */
public final class TrainingReportAccelerationMarkdown {
    private TrainingReportAccelerationMarkdown() {
    }

    public static String render(TrainingReportAcceleration acceleration) {
        if (acceleration == null || !acceleration.available()) {
            return "";
        }
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "## Acceleration");
        appendLine(markdown, "");
        appendLine(markdown, "| Requested | Backend | Device | Accelerated | Fallback | Matmul calls | Matmul delta |");
        appendLine(markdown, "| --- | --- | --- | --- | --- | ---: | ---: |");
        appendLine(markdown, row(acceleration));
        appendLine(markdown, "");
        return markdown.toString();
    }

    private static String row(TrainingReportAcceleration acceleration) {
        return "| `" + escapeTable(acceleration.requestedDevice()) + "`"
                + " | `" + escapeTable(acceleration.executionBackend()) + "`"
                + " | " + escapeTable(acceleration.executionDeviceName())
                + " | `" + yesNo(acceleration.executionAccelerated()) + "`"
                + " | `" + yesNo(acceleration.executionFallback()) + "`"
                + " | " + acceleration.acceleratedMatmulCalls().stream()
                        .mapToObj(String::valueOf)
                        .findFirst()
                        .orElse("")
                + " | " + acceleration.acceleratedMatmulCallsDelta().stream()
                        .mapToObj(String::valueOf)
                        .findFirst()
                        .orElse("")
                + " |";
    }

    private static String yesNo(boolean value) {
        return value ? "yes" : "no";
    }

    private static String escapeTable(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`").replace("|", "\\|").replace("\n", " ");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
