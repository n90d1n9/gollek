package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped cross-run delta dashboard for ranked improvements and regressions.
 */
public record DiffusionOpdGroupedHistoryPortfolioDeltaDashboard(
        DiffusionOpdGroupedHistoryPortfolioComparison comparison,
        int totalEntries,
        int improvedCount,
        int regressedCount,
        int recommendationChangedCount,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> biggestImprovements,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> biggestRegressions,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> recommendationChangedEntries) {
}
