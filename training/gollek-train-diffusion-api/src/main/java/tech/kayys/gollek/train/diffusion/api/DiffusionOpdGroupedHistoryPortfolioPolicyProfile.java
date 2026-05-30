package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed grouped portfolio evaluation under one status policy profile.
 */
public record DiffusionOpdGroupedHistoryPortfolioPolicyProfile(
        String policyName,
        DiffusionOpdRoundHistoryStatusPolicy policy,
        DiffusionOpdGroupedHistoryPortfolioExecutiveSummary executiveSummary,
        DiffusionOpdGroupedHistoryPortfolioRemediationSummary remediationSummary) {
}
