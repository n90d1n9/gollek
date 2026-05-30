package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed downstream bundle for final release delivery artifact state.
 */
public record DiffusionOpdCampaignReleaseDeliveryHandoffBundle(
        DiffusionOpdCampaignReleaseDeliveryReceipt receipt,
        DiffusionOpdCampaignReleaseDeliveryAcknowledgement acknowledgement,
        DiffusionOpdCampaignReleaseDeliveryDecision decision,
        DiffusionOpdCampaignReleaseDeliveryResolution resolution,
        DiffusionOpdCampaignReleaseDeliveryClosure closure,
        String handoffTarget,
        String exportStatus,
        boolean terminal,
        String summaryMessage) {
}
