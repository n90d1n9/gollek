package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired portfolio comparison across two runs.
 */
public record DiffusionOpdPairedHistoryPortfolioComparison(
        String firstField,
        String secondField,
        List<DiffusionOpdPairedHistoryPortfolioComparisonEntry> entries) {
}
