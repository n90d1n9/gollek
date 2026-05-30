package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired portfolio comparison across built-in policy profiles.
 */
public record DiffusionOpdPairedHistoryPortfolioPolicyComparisonSummary(
        List<DiffusionOpdPairedHistoryPortfolioPolicyProfile> profiles,
        String strictestPrimaryAction,
        String loosestPrimaryAction,
        Boolean primaryActionChangedAcrossPolicies,
        int maxReviewHotspotCount,
        int maxManualReviewCount,
        String summaryMessage) {
}
