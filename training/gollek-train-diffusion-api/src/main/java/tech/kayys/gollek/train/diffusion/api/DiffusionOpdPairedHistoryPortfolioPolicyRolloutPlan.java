package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed paired rollout plan derived from multi-policy comparison.
 */
public record DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan(
        DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary comparisonSummary,
        DiffusionOpdPairedHistoryPortfolioPolicyProfile recommendedProfile,
        DiffusionOpdPairedHistoryPortfolioPolicyProfile fallbackProfile,
        String rationale,
        Boolean policyChangedRecommendation) {
}
