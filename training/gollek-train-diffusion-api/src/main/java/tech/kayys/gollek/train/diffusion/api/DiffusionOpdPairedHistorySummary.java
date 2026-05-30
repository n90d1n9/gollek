package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed paired round-history summary for comparisons such as task+teacher or
 * task+stage.
 */
public record DiffusionOpdPairedHistorySummary(
        String firstField,
        String firstValue,
        String secondField,
        String secondValue,
        String pair,
        DiffusionOpdRoundHistorySummary summary) {
}
