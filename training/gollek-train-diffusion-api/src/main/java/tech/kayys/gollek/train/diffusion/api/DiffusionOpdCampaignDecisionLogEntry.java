package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign decision log entry for one orchestration choice.
 */
public record DiffusionOpdCampaignDecisionLogEntry(
        String decisionType,
        String decisionCode,
        String scope,
        String rationale,
        String severity) {
}
