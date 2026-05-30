package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired run-level portfolio for one comparison dimension.
 */
public record DiffusionOpdPairedHistoryPortfolio(
        String firstField,
        String secondField,
        List<DiffusionOpdPairedHistoryDashboardEntry> entries) {
}
