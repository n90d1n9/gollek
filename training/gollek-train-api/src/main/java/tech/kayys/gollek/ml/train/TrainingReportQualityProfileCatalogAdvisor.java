package tech.kayys.gollek.ml.train;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Non-blocking advisor for shaping quality-profile catalogs into maintainable trainer CI workflows.
 */
public final class TrainingReportQualityProfileCatalogAdvisor {
    private TrainingReportQualityProfileCatalogAdvisor() {
    }

    public record Result(
            TrainingReportQualityProfileCatalogValidator.Result validation,
            List<TrainingReportRecommendation> recommendations) {
        public Result {
            validation = validation == null
                    ? new TrainingReportQualityProfileCatalogValidator.Result(false, 0, List.of(), List.of())
                    : validation;
            recommendations = recommendations == null ? List.of() : List.copyOf(recommendations);
        }

        public boolean readyForCi() {
            return validation.passed() && recommendations.stream().noneMatch(TrainingReportRecommendation::blocksPromotion);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("readyForCi", readyForCi());
            map.put("validation", validation.toMap());
            map.put("recommendationCount", recommendations.size());
            map.put("recommendations", recommendations.stream()
                    .map(TrainingReportRecommendation::toMap)
                    .toList());
            return Map.copyOf(map);
        }

        public String markdown() {
            return TrainingReportQualityProfileCatalogAdvisorMarkdown.render(this);
        }

        public static Result fromMap(Map<String, ?> map) {
            Objects.requireNonNull(map, "map must not be null");
            TrainingReportQualityProfileCatalogValidator.Result validation =
                    TrainingReportQualityProfileCatalogValidator.Result.fromMap(
                            TrainingReportValues.mapValue(map, "validation"));
            List<TrainingReportRecommendation> parsedRecommendations = new ArrayList<>();
            Object recommendationsValue = map.get("recommendations");
            if (recommendationsValue instanceof Iterable<?> iterable) {
                for (Object item : iterable) {
                    if (item instanceof Map<?, ?> recommendationMap) {
                        parsedRecommendations.add(TrainingReportRecommendation.fromMap(
                                TrainingReportMapValues.immutableMap(recommendationMap)));
                    }
                }
            }
            return new Result(validation, parsedRecommendations);
        }
    }

    public static Result advise(TrainingReportQualityProfileCatalog catalog) {
        TrainingReportQualityProfileCatalog resolvedCatalog = catalog == null
                ? TrainingReportQualityProfileCatalog.defaults()
                : catalog;
        return advise(resolvedCatalog, resolvedCatalog.validate());
    }

    public static Result advise(Map<String, ?> catalog) {
        TrainingReportQualityProfileCatalogValidator.Result validation =
                TrainingReportQualityProfileCatalogValidator.validate(catalog);
        if (!validation.passed()) {
            return new Result(validation, List.of(validationFailure(validation)));
        }
        return advise(TrainingReportQualityProfileCatalog.fromMap(catalog), validation);
    }

    private static Result advise(
            TrainingReportQualityProfileCatalog catalog,
            TrainingReportQualityProfileCatalogValidator.Result validation) {
        Objects.requireNonNull(catalog, "catalog must not be null");
        List<TrainingReportRecommendation> recommendations = new ArrayList<>();
        addCatalogShapeRecommendations(recommendations, catalog, validation);
        addWorkflowCoverageRecommendations(recommendations, catalog);
        return new Result(validation, recommendations);
    }

    private static void addCatalogShapeRecommendations(
            List<TrainingReportRecommendation> recommendations,
            TrainingReportQualityProfileCatalog catalog,
            TrainingReportQualityProfileCatalogValidator.Result validation) {
        long implicitPolicyWarnings = validation.warnings().stream()
                .filter(issue -> "profile.policy_missing".equals(issue.code()))
                .count();
        if (implicitPolicyWarnings > 0) {
            recommendations.add(new TrainingReportRecommendation(
                    TrainingReportRecommendation.Priority.MEDIUM,
                    TrainingReportRecommendation.Category.REPORTING,
                    TrainingReportDiagnostics.Severity.INFO,
                    "quality_profile_catalog.implicit_policy_defaults",
                    "Make custom quality-profile policy defaults explicit",
                    "One or more profiles rely on implicit Gollek policy defaults, which makes CI behavior harder to review.",
                    List.of(
                            "Declare validationPolicy, performancePolicy, and promotionPolicy on every custom profile.",
                            "Keep permissive local profiles explicit too, so reviewers can distinguish intentional flexibility from missing config.",
                            "Regenerate the catalog validation artifact after policy defaults are made explicit."),
                    Map.of(
                            "implicitPolicyWarningCount", implicitPolicyWarnings,
                            "profileIds", catalog.ids())));
        }
        if (catalog.profiles().size() == 1) {
            recommendations.add(new TrainingReportRecommendation(
                    TrainingReportRecommendation.Priority.MEDIUM,
                    TrainingReportRecommendation.Category.REPORTING,
                    TrainingReportDiagnostics.Severity.INFO,
                    "quality_profile_catalog.single_profile",
                    "Split one-size-fits-all trainer policy into workflow profiles",
                    "A single quality profile usually forces local experiments, research, CI, and production promotion to share the same risk tolerance.",
                    List.of(
                            "Add at least one permissive local/research profile for exploration.",
                            "Add one strict CI profile that fails accelerator fallback and missing validation/report health.",
                            "Add one production promotion profile that requires clean candidate data health."),
                    Map.of("profileIds", catalog.ids())));
        }
    }

    private static void addWorkflowCoverageRecommendations(
            List<TrainingReportRecommendation> recommendations,
            TrainingReportQualityProfileCatalog catalog) {
        if (catalog.profiles().stream().noneMatch(TrainingReportQualityProfileCatalogAdvisor::permissiveLocalProfile)) {
            recommendations.add(new TrainingReportRecommendation(
                    TrainingReportRecommendation.Priority.LOW,
                    TrainingReportRecommendation.Category.REPORTING,
                    TrainingReportDiagnostics.Severity.INFO,
                    "quality_profile_catalog.local_profile_missing",
                    "Add a permissive local experiment profile",
                    "Java training workflows need fast iteration profiles that record diagnostics without blocking exploratory runs.",
                    List.of(
                            "Add a local-experiment profile with permissive validation and performance policy.",
                            "Use the local profile for notebooks, JBang examples, and short trainer smoke runs.",
                            "Keep strict CI and production profiles separate from local experimentation."),
                    Map.of("profileIds", catalog.ids())));
        }
        if (catalog.profiles().stream().noneMatch(TrainingReportQualityProfileCatalogAdvisor::strictAcceleratorCiProfile)) {
            recommendations.add(new TrainingReportRecommendation(
                    TrainingReportRecommendation.Priority.HIGH,
                    TrainingReportRecommendation.Category.OPTIMIZATION,
                    TrainingReportDiagnostics.Severity.WARNING,
                    "quality_profile_catalog.strict_accelerator_ci_missing",
                    "Add a strict accelerator-aware CI profile",
                    "Trainer performance regressions can hide when catalogs do not include a profile that fails accelerator fallback and low throughput.",
                    List.of(
                            "Add a strict-ci profile with failOnAcceleratorFallback enabled.",
                            "Set minTrainSamplesPerSecond to a realistic lower bound for your smallest supported CI model.",
                            "Require fresh diagnostics and checkpoint integrity for this profile."),
                    Map.of("profileIds", catalog.ids())));
        }
        if (catalog.profiles().stream().noneMatch(TrainingReportQualityProfileCatalogAdvisor::productionPromotionProfile)) {
            recommendations.add(new TrainingReportRecommendation(
                    TrainingReportRecommendation.Priority.HIGH,
                    TrainingReportRecommendation.Category.VALIDATION,
                    TrainingReportDiagnostics.Severity.WARNING,
                    "quality_profile_catalog.production_promotion_missing",
                    "Add a production promotion quality profile",
                    "Production model promotion should have a separate profile that is stricter than research and local experimentation.",
                    List.of(
                            "Require clean candidate data health for production promotion.",
                            "Require validation improvement or an explicit promotion-review waiver.",
                            "Keep production promotion thresholds separate from local and research profiles."),
                    Map.of("profileIds", catalog.ids())));
        }
    }

    private static boolean permissiveLocalProfile(TrainingReportQualityProfile profile) {
        return !profile.performancePolicy().failOnAcceleratorFallback()
                || !profile.validationPolicy().requireValidation()
                || !profile.validationPolicy().requireRunHealthGate();
    }

    private static boolean strictAcceleratorCiProfile(TrainingReportQualityProfile profile) {
        return profile.performancePolicy().failOnAcceleratorFallback()
                && profile.performancePolicy().minTrainSamplesPerSecond() >= 1.0
                && profile.validationPolicy().requireFreshDiagnostics()
                && profile.validationPolicy().requireCheckpointIntegrity();
    }

    private static boolean productionPromotionProfile(TrainingReportQualityProfile profile) {
        return profile.promotionPolicy().requireCandidateDataHealthClean()
                && profile.promotionPolicy().requireCandidateDataHealthAvailable();
    }

    private static TrainingReportRecommendation validationFailure(
            TrainingReportQualityProfileCatalogValidator.Result validation) {
        return new TrainingReportRecommendation(
                TrainingReportRecommendation.Priority.BLOCKER,
                TrainingReportRecommendation.Category.REPORTING,
                TrainingReportDiagnostics.Severity.CRITICAL,
                "quality_profile_catalog.validation_failed",
                "Fix quality-profile catalog validation errors before advisory review",
                "Catalog advice is limited until the catalog can be parsed and safely loaded.",
                List.of(
                        "Open the catalog validation Markdown or JUnit XML artifact.",
                        "Fix every validation error, especially duplicate normalized profile ids.",
                        "Rerun catalog validation before using this catalog in CI gates."),
                validation.toMap());
    }
}
