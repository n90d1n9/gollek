package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped cross-run executive summary composed from delta and drift dashboards.
 */
public record DiffusionOpdGroupedHistoryPortfolioExecutiveSummary(
        DiffusionOpdGroupedHistoryPortfolioDeltaDashboard deltaDashboard,
        DiffusionOpdGroupedHistoryPortfolioDriftDashboard driftDashboard,
        DiffusionOpdGroupedHistoryPortfolioComparisonEntry primaryImprovement,
        DiffusionOpdGroupedHistoryPortfolioComparisonEntry primaryRegression,
        DiffusionOpdGroupedHistoryPortfolioComparisonEntry primaryEscalation,
        int reviewHotspotCount,
        List<DiffusionOpdGroupedHistoryPortfolioComparisonEntry> reviewHotspots,
        String summaryMessage) {
}
