package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed final resolution for one release delivery decision.
 */
public record DiffusionOpdCampaignReleaseDeliveryResolution(
        String receiptId,
        String artifactType,
        String destination,
        String resolution,
        String escalationTarget,
        boolean finalAccepted,
        boolean downstreamEligible,
        String summaryMessage) {
}
