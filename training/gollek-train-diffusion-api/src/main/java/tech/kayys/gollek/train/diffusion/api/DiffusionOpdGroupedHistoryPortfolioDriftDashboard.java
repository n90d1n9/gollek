package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped cross-run drift dashboard for status and recommendation changes.
 */
public record DiffusionOpdGroupedHistoryPortfolioDriftDashboard(
        DiffusionOpdGroupedHistoryPortfolioComparison comparison,
        int totalEntries,
        int statusChangedCount,
        int recommendationChangedCount,
        int escalatedCount,
        int healthyToUnhealthyCount,
        int unhealthyToHealthyCount,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> statusChangedEntries,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> recommendationChangedEntries,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> escalatedEntries) {
}
