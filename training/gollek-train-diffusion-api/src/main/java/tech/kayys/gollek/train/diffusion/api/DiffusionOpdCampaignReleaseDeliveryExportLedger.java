package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release-delivery export ledger for downstream delivery export auditing.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportLedger(
        DiffusionOpdCampaignReleaseDeliveryExportReport exportReport,
        List<DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry> entries,
        int entryCount,
        int acknowledgedCount,
        int pendingAcknowledgementCount,
        int retryRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
