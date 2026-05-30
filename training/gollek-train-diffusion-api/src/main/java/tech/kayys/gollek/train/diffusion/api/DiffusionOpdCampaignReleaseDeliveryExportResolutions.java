package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed final resolution collection for release-delivery export decisions.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportResolutions(
        DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions,
        List<DiffusionOpdCampaignReleaseDeliveryExportResolution> resolutions,
        int resolutionCount,
        int acceptedCount,
        int deferredCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
