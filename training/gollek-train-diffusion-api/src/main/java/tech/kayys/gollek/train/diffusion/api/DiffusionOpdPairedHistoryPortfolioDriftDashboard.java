package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired cross-run drift dashboard for status and recommendation changes.
 */
public record DiffusionOpdPairedHistoryPortfolioDriftDashboard(
        DiffusionOpdPairedHistoryPortfolioComparison comparison,
        int totalEntries,
        int statusChangedCount,
        int recommendationChangedCount,
        int escalatedCount,
        int healthyToUnhealthyCount,
        int unhealthyToHealthyCount,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> statusChangedEntries,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> recommendationChangedEntries,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> escalatedEntries) {
}
