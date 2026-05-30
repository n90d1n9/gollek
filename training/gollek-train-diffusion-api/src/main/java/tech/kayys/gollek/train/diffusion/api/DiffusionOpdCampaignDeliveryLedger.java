package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign delivery ledger for downstream delivery auditing.
 */
public record DiffusionOpdCampaignDeliveryLedger(
        DiffusionOpdCampaignDeliveryReport deliveryReport,
        List<DiffusionOpdCampaignDeliveryLedgerEntry> entries,
        int entryCount,
        int acknowledgedCount,
        int pendingAcknowledgementCount,
        int retryRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
