package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed one-call bundle of round-history analysis layers for a slice.
 */
public record DiffusionOpdRoundHistorySnapshot(
        DiffusionOpdRoundHistoryTrend trend,
        DiffusionOpdRoundHistoryStability stability,
        DiffusionOpdRoundHistoryStatus status,
        DiffusionOpdRoundHistoryRecommendation recommendation,
        DiffusionOpdRoundHistoryPlaybook playbook) {
}
