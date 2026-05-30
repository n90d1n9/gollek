package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release delivery ledger entry for one exported release artifact.
 */
public record DiffusionOpdCampaignReleaseDeliveryLedgerEntry(
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
