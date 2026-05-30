package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed recent-window trend view over round-history rows for a slice.
 */
public record DiffusionOpdRoundHistoryTrend(
        int requestedWindowSize,
        int rowCount,
        List<DiffusionOpdRoundHistoryRow> rows,
        DiffusionOpdRoundHistoryRow earliest,
        DiffusionOpdRoundHistoryRow latest,
        Double meanLoss,
        Double minLoss,
        Double maxLoss,
        Double lossDelta,
        Integer roundDelta,
        Boolean improving,
        Boolean worsening) {
}
