package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired remediation bucket for one recommended action.
 */
public record DiffusionOpdPairedHistoryPortfolioRemediationBucket(
        String action,
        int count,
        Boolean reviewRequired,
        Boolean automationFriendly,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> entries) {
}
