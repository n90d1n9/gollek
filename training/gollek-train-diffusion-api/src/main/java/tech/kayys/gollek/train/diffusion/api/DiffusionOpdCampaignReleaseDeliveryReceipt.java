package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release delivery receipt for one exported release artifact handoff.
 */
public record DiffusionOpdCampaignReleaseDeliveryReceipt(
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
