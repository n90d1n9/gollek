package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign export manifest for downstream delivery integrations.
 */
public record DiffusionOpdCampaignExportManifest(
        DiffusionOpdCampaignHandoffBundle handoffBundle,
        List<DiffusionOpdCampaignExportRecord> records,
        int recordCount,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
