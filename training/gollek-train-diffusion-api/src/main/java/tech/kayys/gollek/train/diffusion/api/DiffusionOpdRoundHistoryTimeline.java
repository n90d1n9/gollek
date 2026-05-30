package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed multi-window bundle for comparing short, medium, and long horizons.
 */
public record DiffusionOpdRoundHistoryTimeline(
        int shortWindowSize,
        DiffusionOpdRoundHistorySnapshot shortWindow,
        int mediumWindowSize,
        DiffusionOpdRoundHistorySnapshot mediumWindow,
        int longWindowSize,
        DiffusionOpdRoundHistorySnapshot longWindow,
        Boolean improvingAcrossWindows,
        Boolean stableAcrossWindows,
        Boolean consistentRecommendation,
        Boolean requiresEscalation) {
}
