package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release export manifest for downstream release integrations.
 */
public record DiffusionOpdCampaignReleaseExportManifest(
        DiffusionOpdCampaignReleaseHandoffBundle handoffBundle,
        List<DiffusionOpdCampaignReleaseExportRecord> records,
        int recordCount,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
