package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed acknowledgement collection for release-delivery export receipts.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements(
        DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts,
        List<DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement> acknowledgements,
        int acknowledgementCount,
        int acceptedCount,
        int pendingCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
