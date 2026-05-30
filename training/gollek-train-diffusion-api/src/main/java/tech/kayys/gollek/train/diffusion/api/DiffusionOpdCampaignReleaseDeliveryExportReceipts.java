package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release-delivery export receipt collection for downstream export auditing.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportReceipts(
        DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger,
        List<DiffusionOpdCampaignReleaseDeliveryExportReceipt> receipts,
        int receiptCount,
        int confirmedCount,
        int pendingCount,
        int operatorActionRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
