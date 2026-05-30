package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped remediation summary built from an executive summary.
 */
public record DiffusionOpdGroupedHistoryPortfolioRemediationSummary(
        DiffusionOpdGroupedHistoryPortfolioExecutiveSummary executiveSummary,
        int totalEntries,
        int manualReviewCount,
        int automationFriendlyCount,
        String primaryAction,
        List<DiffusionOpdGroupedHistoryPortfolioRemediationBucket> actionBuckets,
        String summaryMessage) {
}
