package tech.kayys.gollek.ml.train;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Markdown renderer for trainer report quality-profile discovery.
 */
public final class TrainingReportQualityProfileMarkdown {
    private static final DecimalFormat NUMBER_FORMAT = new DecimalFormat(
            "0.######",
            DecimalFormatSymbols.getInstance(Locale.ROOT));

    private TrainingReportQualityProfileMarkdown() {
    }

    public static String render() {
        return render(TrainingReportQualityProfileCatalog.defaults());
    }

    public static String render(List<TrainingReportQualityProfile> profiles) {
        return render(new TrainingReportQualityProfileCatalog(profiles));
    }

    public static String render(TrainingReportQualityProfileCatalog catalog) {
        Objects.requireNonNull(catalog, "catalog must not be null");
        StringBuilder markdown = new StringBuilder();
        appendLine(markdown, "# Gollek Training Report Quality Profiles");
        appendLine(markdown, "");
        appendLine(markdown, "Use these named profiles to keep local experiments, CI validation, and production promotion gates consistent.");
        appendLine(markdown, "");
        appendLine(markdown, "| Profile | Purpose | Validation Policy | Performance Policy | Promotion Policy |");
        appendLine(markdown, "| --- | --- | --- | --- | --- |");
        for (TrainingReportQualityProfile profile : catalog.profiles()) {
            appendLine(markdown, row(profile));
        }
        return markdown.toString();
    }

    private static String row(TrainingReportQualityProfile profile) {
        return "| `" + escapeTable(profile.id()) + "`"
                + " | " + escapeTable(profile.description())
                + " | " + validationPolicy(profile.validationPolicy())
                + " | " + performancePolicy(profile.performancePolicy())
                + " | " + promotionPolicy(profile.promotionPolicy())
                + " |";
    }

    private static String validationPolicy(TrainingReportValidationPolicy policy) {
        return "diagnostics <= `" + policy.maxDiagnosticSeverity().name() + "`"
                + ", run health gate `" + policy.requireRunHealthGate() + "`"
                + ", data health gate `" + policy.requireDataHealthGate() + "`"
                + ", data health evidence `" + policy.requireDataHealthAvailable() + "`"
                + ", fresh diagnostics `" + policy.requireFreshDiagnostics() + "`"
                + ", validation `" + policy.requireValidation() + "`"
                + ", checkpoint integrity `" + policy.requireCheckpointIntegrity() + "`";
    }

    private static String promotionPolicy(TrainingReportPortfolio.PromotionPolicy policy) {
        return "candidate diagnostics <= `" + policy.maxCandidateDiagnosticSeverity().name() + "`"
                + ", comparison <= `" + policy.maxComparisonFindingSeverity().name() + "`"
                + ", min validation improvement `" + formatNumber(policy.minimumValidationImprovement()) + "`"
                + ", tracked metric `" + policy.requireTrackedMetricImprovement() + "`"
                + ", data health evidence `" + policy.requireCandidateDataHealthAvailable() + "`"
                + ", data health gate `" + policy.requireCandidateDataHealthGate() + "`"
                + ", clean data health `" + policy.requireCandidateDataHealthClean() + "`";
    }

    private static String performancePolicy(TrainingReportPerformanceGate.Policy policy) {
        return "accelerator fallback fails `" + policy.failOnAcceleratorFallback() + "`"
                + ", min train samples/s `" + formatNumber(policy.minTrainSamplesPerSecond()) + "`"
                + ", max validation/train batch ms ratio `"
                + formatNumber(policy.maxValidationToTrainAverageBatchMillisRatio()) + "`";
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
