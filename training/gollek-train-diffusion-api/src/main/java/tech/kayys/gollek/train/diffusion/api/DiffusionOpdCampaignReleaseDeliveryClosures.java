package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed terminal closure collection for release delivery resolutions.
 */
public record DiffusionOpdCampaignReleaseDeliveryClosures(
        DiffusionOpdCampaignReleaseDeliveryResolutions deliveryResolutions,
        List<DiffusionOpdCampaignReleaseDeliveryClosure> closures,
        int closureCount,
        int closedCount,
        int openCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
