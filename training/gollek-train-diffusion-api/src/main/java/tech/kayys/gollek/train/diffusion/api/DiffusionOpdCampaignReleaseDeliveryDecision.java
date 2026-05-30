package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed delivery decision for one release delivery acknowledgement.
 */
public record DiffusionOpdCampaignReleaseDeliveryDecision(
        String receiptId,
        String artifactType,
        String destination,
        String decision,
        String reviewer,
        String reviewerRole,
        String resolutionRoute,
        boolean finalDecision,
        String summaryMessage) {
}
