package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release-delivery export receipt for one exported handoff artifact.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportReceipt(
        String artifactType,
        String destination,
        String referenceKey,
        String receiptId,
        String receiptStatus,
        String acknowledgementStatus,
        String operatorAcknowledgement,
        String deliveryTimestamp,
        boolean followUpRequired,
        String summaryMessage) {
}
