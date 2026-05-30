package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed paired dashboard entry with summary and richer slice analysis.
 */
public record DiffusionOpdPairedHistoryDashboardEntry(
        String firstField,
        String firstValue,
        String secondField,
        String secondValue,
        String pair,
        DiffusionOpdRoundHistorySummary summary,
        DiffusionOpdRoundHistorySnapshot snapshot,
        DiffusionOpdRoundHistoryTimeline timeline) {
}
