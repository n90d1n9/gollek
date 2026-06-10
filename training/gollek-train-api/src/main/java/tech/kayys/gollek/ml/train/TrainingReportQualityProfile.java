package tech.kayys.gollek.ml.train;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Named validation and promotion policy bundles for common trainer workflows.
 */
public record TrainingReportQualityProfile(
        String id,
        String displayName,
        String description,
        TrainingReportValidationPolicy validationPolicy,
        TrainingReportPerformanceGate.Policy performancePolicy,
        TrainingReportPortfolio.PromotionPolicy promotionPolicy) {
    public static final String LOCAL_EXPERIMENT = "local-experiment";
    public static final String RESEARCH = "research";
    public static final String STRICT_CI = "strict-ci";
    public static final String PRODUCTION_PROMOTION = "production-promotion";

    public TrainingReportQualityProfile {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        id = normalize(id);
        displayName = displayName == null || displayName.isBlank() ? id : displayName.trim();
        description = description == null ? "" : description.trim();
        validationPolicy = validationPolicy == null
                ? TrainingReportValidationPolicy.defaultPolicy()
                : validationPolicy;
        performancePolicy = performancePolicy == null
                ? TrainingReportPerformanceGate.Policy.defaults()
                : performancePolicy;
        promotionPolicy = promotionPolicy == null
                ? TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                : promotionPolicy;
    }

    public static TrainingReportQualityProfile localExperiment() {
        return new TrainingReportQualityProfile(
                LOCAL_EXPERIMENT,
                "Local Experiment",
                "Fast local iteration that records quality signals without blocking exploratory runs.",
                TrainingReportValidationPolicy.permissive()
                        .withRequireDataHealthGate(false),
                TrainingReportPerformanceGate.Policy.permissive(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.CRITICAL)
                        .withMaxComparisonFindingSeverity(TrainingReportDiagnostics.Severity.CRITICAL)
                        .withMinimumValidationImprovement(0.0)
                        .withRequireTrackedMetricImprovement(false)
                        .withRequireCandidateDataHealthGate(false));
    }

    public static TrainingReportQualityProfile research() {
        return new TrainingReportQualityProfile(
                RESEARCH,
                "Research",
                "Balanced research workflow that tolerates warnings while still blocking failed health gates.",
                new TrainingReportValidationPolicy(
                        TrainingReportDiagnostics.Severity.WARNING,
                        true,
                        true,
                        false,
                        false,
                        true,
                        false),
                TrainingReportPerformanceGate.Policy.defaults(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withMaxComparisonFindingSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withRequireTrackedMetricImprovement(false));
    }

    public static TrainingReportQualityProfile strictCi() {
        return new TrainingReportQualityProfile(
                STRICT_CI,
                "Strict CI",
                "CI gate that requires fresh validation, run health, data-health evidence, and checkpoint integrity.",
                TrainingReportValidationPolicy.strict()
                        .withRequireDataHealthAvailable(true),
                TrainingReportPerformanceGate.Policy.strict(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withRequireCandidateDataHealthAvailable(true));
    }

    public static TrainingReportQualityProfile productionPromotion() {
        return new TrainingReportQualityProfile(
                PRODUCTION_PROMOTION,
                "Production Promotion",
                "Production promotion gate that turns warnings into review holds and requires clean candidate data health.",
                TrainingReportValidationPolicy.strict()
                        .withRequireDataHealthAvailable(true),
                TrainingReportPerformanceGate.Policy.strict(),
                TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                        .withMaxCandidateDiagnosticSeverity(TrainingReportDiagnostics.Severity.WARNING)
                        .withRequireCandidateDataHealthAvailable(true)
                        .withRequireCandidateDataHealthClean(true));
    }

    public static List<TrainingReportQualityProfile> defaults() {
        return List.of(
                localExperiment(),
                research(),
                strictCi(),
                productionPromotion());
    }

    public static Optional<TrainingReportQualityProfile> find(String id) {
        if (id == null || id.isBlank()) {
            return Optional.empty();
        }
        String normalized = normalize(id);
        return defaults().stream()
                .filter(profile -> profile.id().equals(normalized))
                .findFirst();
    }

    public static TrainingReportQualityProfile require(String id) {
        return find(id)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Unknown training report quality profile: " + id
                                + ". Available profiles: " + ids()));
    }

    public static TrainingReportQualityProfile fromMap(Map<String, ?> map) {
        Objects.requireNonNull(map, "map must not be null");
        String id = TrainingReportValues.stringValue(map.get("id"), "");
        if (id.isBlank()) {
            throw new IllegalArgumentException("Training report quality profile id must not be blank");
        }
        return new TrainingReportQualityProfile(
                id,
                TrainingReportValues.stringValue(map.get("displayName"), id),
                TrainingReportValues.stringValue(map.get("description"), ""),
                TrainingReportValidationPolicy.fromMap(
                        TrainingReportValues.mapValue(map, "validationPolicy")),
                TrainingReportPerformanceGate.Policy.fromMap(
                        TrainingReportValues.mapValue(map, "performancePolicy")),
                TrainingReportPortfolio.PromotionPolicy.fromMap(
                        TrainingReportValues.mapValue(map, "promotionPolicy")));
    }

    public static List<String> ids() {
        return defaults().stream()
                .map(TrainingReportQualityProfile::id)
                .toList();
    }

    public TrainingReportValidationPolicy.Result validate(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return report.validate(validationPolicy);
    }

    public TrainingReportPerformanceGate.Result performanceGate(TrainingReport report) {
        Objects.requireNonNull(report, "report must not be null");
        return report.performanceGate(performancePolicy);
    }

    public TrainingReportPortfolio.PromotionReview promotionReview(
            TrainingReportPortfolio portfolio,
            String baselineName) {
        Objects.requireNonNull(portfolio, "portfolio must not be null");
        return portfolio.promotionReview(baselineName, promotionPolicy);
    }

    public TrainingReportPortfolio.PromotionDecision promotionDecision(
            TrainingReportPortfolio portfolio,
            String baselineName) {
        return promotionReview(portfolio, baselineName).decision();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("id", id);
        map.put("displayName", displayName);
        map.put("description", description);
        map.put("validationPolicy", validationPolicy.toMap());
        map.put("performancePolicy", performancePolicy.toMap());
        map.put("promotionPolicy", promotionPolicy.toMap());
        return Map.copyOf(map);
    }

    static String normalizeId(String id) {
        return id.trim()
                .toLowerCase(Locale.ROOT)
                .replace('_', '-')
                .replaceAll("\\s+", "-");
    }

    private static String normalize(String id) {
        return normalizeId(id);
    }
}
