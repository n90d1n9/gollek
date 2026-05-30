package tech.kayys.gollek.ml.train;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * CI-friendly Markdown renderer for training promotion reviews.
 */
public final class TrainingReportPromotionMarkdown {
    private static final DecimalFormat NUMBER_FORMAT = new DecimalFormat(
            "0.######",
            DecimalFormatSymbols.getInstance(Locale.ROOT));

    private TrainingReportPromotionMarkdown() {
    }

    public static String render(TrainingReportPortfolio.PromotionReview review) {
        Objects.requireNonNull(review, "review must not be null");
        TrainingReportPortfolio.PromotionDecision decision = review.decision();
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Promotion Review");
        appendLine(markdown, "");
        appendLine(markdown, "**Decision:** " + renderDecision(decision));
        appendLine(markdown, "");
        appendLine(markdown, "Baseline: `" + escapeInline(review.baseline().name()) + "`");
        appendLine(markdown, "Policy: candidate diagnostics <= `"
                + review.policy().maxCandidateDiagnosticSeverity().name()
                + "`, comparison findings <= `"
                + review.policy().maxComparisonFindingSeverity().name()
                + "`, minimum validation improvement `"
                + formatNumber(review.policy().minimumValidationImprovement())
                + "`.");
        appendLine(markdown, "");
        appendSourceReports(markdown, review);
        appendLine(markdown, "| Candidate | Status | Validation score | Diagnostics | Improved | Regressed | Reasons |");
        appendLine(markdown, "| --- | --- | ---: | --- | --- | --- | --- |");
        for (TrainingReportPortfolio.PromotionCandidateReview candidate : review.candidates()) {
            appendLine(markdown, row(candidate));
        }
        return markdown.toString();
    }

    public static String renderDecision(TrainingReportPortfolio.PromotionDecision decision) {
        Objects.requireNonNull(decision, "decision must not be null");
        if (decision.promotable()) {
            return "`PROMOTE` candidate `" + escapeInline(decision.candidate()
                    .map(TrainingReportPortfolio.Entry::name)
                    .orElse("<none>"))
                    + "` over baseline `" + escapeInline(decision.baseline().name()) + "`.";
        }
        if (decision.status() == TrainingReportPortfolio.PromotionStatus.NO_ELIGIBLE_CANDIDATE) {
            return "`NO_ELIGIBLE_CANDIDATE` for baseline `" + escapeInline(decision.baseline().name())
                    + "` because " + reasons(decision.reasons()) + ".";
        }
        return "`HOLD` baseline `" + escapeInline(decision.baseline().name())
                + "` against candidate `" + escapeInline(decision.candidate()
                        .map(TrainingReportPortfolio.Entry::name)
                        .orElse("<none>"))
                + "` because " + reasons(decision.reasons()) + ".";
    }

    private static String row(TrainingReportPortfolio.PromotionCandidateReview candidate) {
        TrainingReportComparison comparison = candidate.comparison();
        return "| `" + escapeTable(candidate.candidate().name()) + "`"
                + " | `" + candidate.status().name() + "`"
                + " | " + validationScore(candidate.candidate())
                + " | " + diagnostics(candidate)
                + " | " + metricNames(comparison.improvedMetrics())
                + " | " + metricNames(comparison.regressedMetrics())
                + " | " + escapeTable(reasons(candidate.reasons()))
                + " |";
    }

    private static void appendSourceReports(
            StringBuilder markdown,
            TrainingReportPortfolio.PromotionReview review) {
        if (!hasSourceProvenance(review)) {
            return;
        }
        appendLine(markdown, "## Source Reports");
        appendLine(markdown, "");
        appendLine(markdown, "| Role | Name | Source | Bytes | SHA-256 |");
        appendLine(markdown, "| --- | --- | --- | ---: | --- |");
        appendLine(markdown, sourceRow("baseline", review.baseline()));
        for (TrainingReportPortfolio.PromotionCandidateReview candidate : review.candidates()) {
            appendLine(markdown, sourceRow("candidate", candidate.candidate()));
        }
        appendLine(markdown, "");
    }

    private static boolean hasSourceProvenance(TrainingReportPortfolio.PromotionReview review) {
        if (hasSourceProvenance(review.baseline())) {
            return true;
        }
        for (TrainingReportPortfolio.PromotionCandidateReview candidate : review.candidates()) {
            if (hasSourceProvenance(candidate.candidate())) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasSourceProvenance(TrainingReportPortfolio.Entry entry) {
        return entry.source() != null || entry.sourceBytes() != null || entry.sourceSha256() != null;
    }

    private static String sourceRow(String role, TrainingReportPortfolio.Entry entry) {
        return "| `" + escapeTable(role) + "`"
                + " | `" + escapeTable(entry.name()) + "`"
                + " | " + sourcePath(entry)
                + " | " + sourceBytes(entry)
                + " | " + sourceSha256(entry)
                + " |";
    }

    private static String validationScore(TrainingReportPortfolio.Entry entry) {
        OptionalDouble score = entry.validationScore();
        return score.isPresent() ? formatNumber(score.getAsDouble()) : "n/a";
    }

    private static String diagnostics(TrainingReportPortfolio.PromotionCandidateReview candidate) {
        String severity = candidate.candidate().report().highestDiagnosticSeverity();
        String result = candidate.diagnosticsPassed() ? "pass" : "fail";
        return "`" + escapeTable(result) + "` / `" + escapeTable(severity) + "`";
    }

    private static String metricNames(List<TrainingReportComparison.MetricDelta> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return "-";
        }
        List<String> names = new ArrayList<>(metrics.size());
        for (TrainingReportComparison.MetricDelta metric : metrics) {
            names.add("`" + escapeTable(metric.name()) + "`");
        }
        return String.join(", ", names);
    }

    private static String sourcePath(TrainingReportPortfolio.Entry entry) {
        return entry.source() == null ? "-" : "`" + escapeTable(entry.source().toString()) + "`";
    }

    private static String sourceBytes(TrainingReportPortfolio.Entry entry) {
        return entry.sourceBytes() == null ? "-" : Long.toString(entry.sourceBytes());
    }

    private static String sourceSha256(TrainingReportPortfolio.Entry entry) {
        return entry.sourceSha256() == null ? "-" : "`" + escapeTable(entry.sourceSha256()) + "`";
    }

    private static String reasons(List<String> reasons) {
        if (reasons == null || reasons.isEmpty()) {
            return "none";
        }
        return String.join("; ", reasons);
    }

    private static String formatNumber(double value) {
        if (!Double.isFinite(value)) {
            return "n/a";
        }
        synchronized (NUMBER_FORMAT) {
            return NUMBER_FORMAT.format(value);
        }
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
