package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired cross-run executive summary composed from delta and drift dashboards.
 */
public record DiffusionOpdPairedHistoryPortfolioExecutiveSummary(
        DiffusionOpdPairedHistoryPortfolioDeltaDashboard deltaDashboard,
        DiffusionOpdPairedHistoryPortfolioDriftDashboard driftDashboard,
        DiffusionOpdPairedHistoryPortfolioComparisonEntry primaryImprovement,
        DiffusionOpdPairedHistoryPortfolioComparisonEntry primaryRegression,
        DiffusionOpdPairedHistoryPortfolioComparisonEntry primaryEscalation,
        int reviewHotspotCount,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> reviewHotspots,
        String summaryMessage) {
}
