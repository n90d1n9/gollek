package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed downstream bundle for final release-delivery export artifact state.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle(
        DiffusionOpdCampaignReleaseDeliveryExportReceipt receipt,
        DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement acknowledgement,
        DiffusionOpdCampaignReleaseDeliveryExportDecision decision,
        DiffusionOpdCampaignReleaseDeliveryExportResolution resolution,
        DiffusionOpdCampaignReleaseDeliveryExportClosure closure,
        String handoffTarget,
        String exportStatus,
        boolean terminal,
        String summaryMessage) {
}
