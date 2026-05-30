package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed delta between the latest and previous round-history rows for a slice.
 */
public record DiffusionOpdRoundHistoryDelta(
        DiffusionOpdRoundHistoryRow latest,
        DiffusionOpdRoundHistoryRow previous,
        Double lossDelta,
        Integer roundDelta) {
}
