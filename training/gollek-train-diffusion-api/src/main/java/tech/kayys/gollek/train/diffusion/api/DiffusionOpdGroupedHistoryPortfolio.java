package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped run-level portfolio for one logical dimension.
 */
public record DiffusionOpdGroupedHistoryPortfolio(
        String field,
        List<DiffusionOpdGroupedHistoryDashboardEntry> entries) {
}
