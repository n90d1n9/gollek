package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Compact Markdown renderer for promotion gate results.
 */
public final class TrainingReportPromotionGateMarkdown {
    private TrainingReportPromotionGateMarkdown() {
    }

    public static String render(TrainingReportPromotionGate.Result result) {
        Objects.requireNonNull(result, "result must not be null");
        TrainingReportPromotionArtifacts.ArtifactBundle artifacts = result.artifacts();
        TrainingReportPromotionArtifacts.ArtifactVerification verification = result.verification();
        TrainingReportPromotionArtifacts.SourceVerification sourceVerification = result.sourceVerification();

        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Promotion Gate");
        appendLine(markdown, "");
        appendLine(markdown, "**Gate:** `" + (result.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Decision:** "
                + TrainingReportPromotionMarkdown.renderDecision(result.decision()));
        appendLine(markdown, "**Artifact verification:** `"
                + (verification.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "**Source report verification:** `"
                + (sourceVerification.passed() ? "PASS" : "FAIL") + "`");
        appendLine(markdown, "");
        appendLine(markdown, "## Artifacts");
        appendLine(markdown, "");
        appendLine(markdown, "| Artifact | Path | SHA-256 | Verified |");
        appendLine(markdown, "| --- | --- | --- | --- |");
        appendLine(markdown, artifactRow(
                "JSON review",
                artifacts.jsonFile(),
                artifacts.jsonSha256(),
                verification.jsonSha256Matches()));
        appendLine(markdown, artifactRow(
                "Markdown review",
                artifacts.markdownFile(),
                artifacts.markdownSha256(),
                verification.markdownSha256Matches()));
        if (!sourceVerification.reports().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Source Reports");
            appendLine(markdown, "");
            appendLine(markdown, "| Role | Name | Path | Bytes | SHA-256 |");
            appendLine(markdown, "| --- | --- | --- | ---: | --- |");
            for (TrainingReportPromotionArtifacts.SourceReport report : sourceVerification.reports()) {
                appendLine(markdown, sourceReportRow(report));
            }
        }
        appendLine(markdown, "");
        appendLine(markdown, "## Review");
        appendLine(markdown, "");
        appendLine(markdown, "Baseline: `" + escapeInline(result.review().baseline().name()) + "`");
        appendLine(markdown, "Candidates audited: `" + result.review().candidates().size() + "`");
        appendLine(markdown, "Promotable candidates: `" + result.review().promotableCandidates().size() + "`");
        appendLine(markdown, "Held candidates: `" + result.review().heldCandidates().size() + "`");
        appendLine(markdown, "");
        appendLine(markdown, result.message());
        if (!verification.failures().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Verification Failures");
            appendLine(markdown, "");
            for (String failure : verification.failures()) {
                appendLine(markdown, "- " + escapeListItem(failure));
            }
        }
        if (!sourceVerification.failures().isEmpty()) {
            appendLine(markdown, "");
            appendLine(markdown, "## Source Verification Failures");
            appendLine(markdown, "");
            for (String failure : sourceVerification.failures()) {
                appendLine(markdown, "- " + escapeListItem(failure));
            }
        }
        return markdown.toString();
    }

    private static String artifactRow(
            String label,
            Path path,
            String sha256,
            boolean verified) {
        return "| " + escapeTable(label)
                + " | `" + escapeTable(path.toString()) + "`"
                + " | `" + escapeTable(shortSha(sha256)) + "`"
                + " | `" + (verified ? "yes" : "no") + "` |";
    }

    private static String sourceReportRow(TrainingReportPromotionArtifacts.SourceReport report) {
        String path = report.source() == null ? "n/a" : report.source().toString();
        String bytes = report.bytes() == null ? "n/a" : Long.toString(report.bytes());
        return "| " + escapeTable(report.role())
                + " | `" + escapeTable(report.name()) + "`"
                + " | `" + escapeTable(path) + "`"
                + " | `" + escapeTable(bytes) + "`"
                + " | `" + escapeTable(shortSha(report.sha256())) + "` |";
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

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
