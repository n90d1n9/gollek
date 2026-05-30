package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed recent-window stability view layered on top of a trend window.
 */
public record DiffusionOpdRoundHistoryStability(
        DiffusionOpdRoundHistoryTrend trend,
        Double lossRange,
        Double averageSignedStepDelta,
        Double averageAbsoluteStepDelta,
        Double latestAbsoluteStepDelta,
        Boolean volatileWindow,
        Boolean stabilizing) {
}
