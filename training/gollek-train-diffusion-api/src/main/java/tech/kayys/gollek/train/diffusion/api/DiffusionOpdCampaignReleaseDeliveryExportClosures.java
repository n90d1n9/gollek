package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed terminal closure collection for release-delivery export resolutions.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportClosures(
        DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions,
        List<DiffusionOpdCampaignReleaseDeliveryExportClosure> closures,
        int closureCount,
        int closedCount,
        int openCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
