package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release-delivery export manifest for downstream delivery integrations.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportManifest(
        List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> handoffBundles,
        List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records,
        int recordCount,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
