package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed next-action recommendation layered on top of round-history status.
 */
public record DiffusionOpdRoundHistoryRecommendation(
        DiffusionOpdRoundHistoryStatus status,
        String action,
        String priority,
        String reasonCode,
        String rationale,
        Boolean automationFriendly) {
}
