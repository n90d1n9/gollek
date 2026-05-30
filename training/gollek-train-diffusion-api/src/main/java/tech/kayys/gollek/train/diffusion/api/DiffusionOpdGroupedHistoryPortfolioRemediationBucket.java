package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped remediation bucket for one recommended action.
 */
public record DiffusionOpdGroupedHistoryPortfolioRemediationBucket(
        String action,
        int count,
        Boolean reviewRequired,
        Boolean automationFriendly,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> entries) {
}
