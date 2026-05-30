package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release delivery ledger for downstream release delivery auditing.
 */
public record DiffusionOpdCampaignReleaseDeliveryLedger(
        DiffusionOpdCampaignReleaseDeliveryReport deliveryReport,
        List<DiffusionOpdCampaignReleaseDeliveryLedgerEntry> entries,
        int entryCount,
        int acknowledgedCount,
        int pendingAcknowledgementCount,
        int retryRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
