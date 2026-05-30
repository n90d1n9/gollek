package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign export record for one delivered artifact.
 */
public record DiffusionOpdCampaignExportRecord(
        String artifactType,
        String destination,
        String deliveryStatus,
        String referenceKey) {
}
