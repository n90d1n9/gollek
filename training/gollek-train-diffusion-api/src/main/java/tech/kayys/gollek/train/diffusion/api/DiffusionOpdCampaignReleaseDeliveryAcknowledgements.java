package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed acknowledgement collection for release delivery receipts.
 */
public record DiffusionOpdCampaignReleaseDeliveryAcknowledgements(
        DiffusionOpdCampaignReleaseDeliveryReceipts deliveryReceipts,
        List<DiffusionOpdCampaignReleaseDeliveryAcknowledgement> acknowledgements,
        int acknowledgementCount,
        int acceptedCount,
        int pendingCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
