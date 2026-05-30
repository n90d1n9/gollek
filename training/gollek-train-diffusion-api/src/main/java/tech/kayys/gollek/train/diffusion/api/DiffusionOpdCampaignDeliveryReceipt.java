package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign delivery receipt for one exported artifact handoff.
 */
public record DiffusionOpdCampaignDeliveryReceipt(
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
