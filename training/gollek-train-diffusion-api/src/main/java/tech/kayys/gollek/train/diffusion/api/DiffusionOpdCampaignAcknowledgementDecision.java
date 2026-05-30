package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed decision record for one workflow campaign receipt acknowledgement.
 */
public record DiffusionOpdCampaignAcknowledgementDecision(
        String receiptId,
        String artifactType,
        String decision,
        String reviewer,
        String reviewerRole,
        String note,
        String resolutionRoute,
        boolean finalDecision,
        String summaryMessage) {
}
