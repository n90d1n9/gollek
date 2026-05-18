package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed summary view of a DiffusionOPD round-history slice.
 */
public record DiffusionOpdRoundHistorySummary(
        int count,
        Double meanLoss,
        DiffusionOpdRoundHistoryRow last,
        List<DiffusionOpdRoundHistoryRow> topLosses,
        List<DiffusionOpdRoundHistoryRow> firstRounds) {
}
