package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign delivery ledger entry for one exported artifact.
 */
public record DiffusionOpdCampaignDeliveryLedgerEntry(
        String artifactType,
        String destination,
        String referenceKey,
        String deliveryStatus,
        String acknowledgementStatus,
        String retryPolicy,
        int attemptCount,
        boolean followUpRequired,
        String summaryMessage) {
}
