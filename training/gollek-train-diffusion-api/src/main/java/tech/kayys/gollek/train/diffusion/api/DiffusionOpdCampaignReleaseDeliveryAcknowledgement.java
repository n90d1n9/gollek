package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed acknowledgement record for one release delivery receipt.
 */
public record DiffusionOpdCampaignReleaseDeliveryAcknowledgement(
        String receiptId,
        String artifactType,
        String destination,
        String acknowledgementOutcome,
        String reviewer,
        String reviewerRole,
        boolean finalAcknowledgement,
        boolean followUpRequired,
        String summaryMessage) {
}
