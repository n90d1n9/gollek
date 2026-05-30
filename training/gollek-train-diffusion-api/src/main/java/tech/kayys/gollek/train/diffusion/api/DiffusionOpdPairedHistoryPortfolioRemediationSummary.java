package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired remediation summary built from an executive summary.
 */
public record DiffusionOpdPairedHistoryPortfolioRemediationSummary(
        DiffusionOpdPairedHistoryPortfolioExecutiveSummary executiveSummary,
        int totalEntries,
        int manualReviewCount,
        int automationFriendlyCount,
        String primaryAction,
        List<DiffusionOpdPairedHistoryPortfolioRemediationBucket> actionBuckets,
        String summaryMessage) {
}
