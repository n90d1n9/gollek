package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed grouped round-history summary for one logical value such as a task,
 * teacher, or stage.
 */
public record DiffusionOpdGroupedRoundHistorySummary(
        String field,
        String value,
        DiffusionOpdRoundHistorySummary summary) {
}
