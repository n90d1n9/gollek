package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release-delivery export record for one downstream handoff artifact.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportRecord(
        String artifactType,
        String destination,
        String deliveryStatus,
        String referenceKey) {
}
