package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed final resolution for one release-delivery export decision.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportResolution(
        String receiptId,
        String artifactType,
        String destination,
        String resolution,
        String escalationTarget,
        boolean finalAccepted,
        boolean downstreamEligible,
        String summaryMessage) {
}
