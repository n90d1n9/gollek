package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed paired portfolio comparison entry for one pair value.
 */
public record DiffusionOpdPairedHistoryPortfolioComparisonEntry(
        String firstField,
        String firstValue,
        String secondField,
        String secondValue,
        String pair,
        DiffusionOpdPairedHistoryDashboardEntry baseline,
        DiffusionOpdPairedHistoryDashboardEntry current,
        Double meanLossDelta,
        Boolean improved,
        Boolean statusChanged,
        Boolean recommendationChanged) {
}
