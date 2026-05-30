package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed acknowledgement collection for workflow campaign delivery receipts.
 */
public record DiffusionOpdCampaignReceiptAcknowledgements(
        DiffusionOpdCampaignDeliveryReceipts deliveryReceipts,
        List<DiffusionOpdCampaignReceiptAcknowledgement> acknowledgements,
        int acknowledgementCount,
        int acceptedCount,
        int pendingCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
