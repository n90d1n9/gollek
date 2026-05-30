package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed grouped portfolio comparison entry for one logical value.
 */
public record DiffusionOpdGroupedHistoryPortfolioComparisonEntry(
        String field,
        String value,
        DiffusionOpdGroupedHistoryDashboardEntry baseline,
        DiffusionOpdGroupedHistoryDashboardEntry current,
        Double meanLossDelta,
        Boolean improved,
        Boolean statusChanged,
        Boolean recommendationChanged) {
}
