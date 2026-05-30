package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired cross-run delta dashboard for ranked improvements and regressions.
 */
public record DiffusionOpdPairedHistoryPortfolioDeltaDashboard(
        DiffusionOpdPairedHistoryPortfolioComparison comparison,
        int totalEntries,
        int improvedCount,
        int regressedCount,
        int recommendationChangedCount,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> biggestImprovements,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> biggestRegressions,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> recommendationChangedEntries) {
}
