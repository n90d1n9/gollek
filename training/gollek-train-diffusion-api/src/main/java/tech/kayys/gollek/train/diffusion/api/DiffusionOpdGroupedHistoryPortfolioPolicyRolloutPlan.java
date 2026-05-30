package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed grouped rollout plan derived from multi-policy comparison.
 */
public record DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan(
        DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary comparisonSummary,
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile recommendedProfile,
        DiffusionOpdGroupedHistoryPortfolioPolicyProfile fallbackProfile,
        String rationale,
        Boolean policyChangedRecommendation) {
}
