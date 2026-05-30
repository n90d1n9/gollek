package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped portfolio comparison across built-in policy profiles.
 */
public record DiffusionOpdGroupedHistoryPortfolioPolicyComparisonSummary(
        List<DiffusionOpdGroupedHistoryPortfolioPolicyProfile> profiles,
        String strictestPrimaryAction,
        String loosestPrimaryAction,
        Boolean primaryActionChangedAcrossPolicies,
        int maxReviewHotspotCount,
        int maxManualReviewCount,
        String summaryMessage) {
}
