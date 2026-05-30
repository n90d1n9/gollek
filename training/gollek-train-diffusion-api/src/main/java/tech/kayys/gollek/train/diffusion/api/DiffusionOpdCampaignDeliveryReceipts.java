package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign delivery receipt collection for downstream auditing.
 */
public record DiffusionOpdCampaignDeliveryReceipts(
        DiffusionOpdCampaignDeliveryLedger deliveryLedger,
        List<DiffusionOpdCampaignDeliveryReceipt> receipts,
        int receiptCount,
        int confirmedCount,
        int pendingCount,
        int operatorActionRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
