package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed alert/status view layered on top of round-history stability analysis.
 */
public record DiffusionOpdRoundHistoryStatus(
        DiffusionOpdRoundHistoryStability stability,
        String status,
        String alertLevel,
        String issueCode,
        String summaryMessage,
        Boolean healthy,
        Boolean regressing,
        Boolean unstable) {
}
