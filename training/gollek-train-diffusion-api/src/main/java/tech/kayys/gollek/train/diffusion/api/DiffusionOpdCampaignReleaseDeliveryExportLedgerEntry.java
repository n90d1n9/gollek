package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release-delivery export ledger entry for one exported handoff artifact.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry(
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
