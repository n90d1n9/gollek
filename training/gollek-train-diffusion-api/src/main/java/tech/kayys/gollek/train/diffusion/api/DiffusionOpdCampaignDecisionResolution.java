package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed final resolution for one campaign acknowledgement decision.
 */
public record DiffusionOpdCampaignDecisionResolution(
        String receiptId,
        String artifactType,
        String resolution,
        String escalationOwner,
        String executionReadiness,
        String downstreamEligibility,
        boolean finalResolution,
        String summaryMessage) {
}
