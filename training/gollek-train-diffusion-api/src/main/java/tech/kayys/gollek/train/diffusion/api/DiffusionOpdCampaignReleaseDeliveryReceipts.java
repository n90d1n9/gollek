package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release delivery receipt collection for downstream release auditing.
 */
public record DiffusionOpdCampaignReleaseDeliveryReceipts(
        DiffusionOpdCampaignReleaseDeliveryLedger deliveryLedger,
        List<DiffusionOpdCampaignReleaseDeliveryReceipt> receipts,
        int receiptCount,
        int confirmedCount,
        int pendingCount,
        int operatorActionRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
