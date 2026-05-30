package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed destination-grouped release-delivery export report.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportReport(
        DiffusionOpdCampaignReleaseDeliveryExportManifest exportManifest,
        List<DiffusionOpdCampaignReleaseDeliveryDestination> destinations,
        int destinationCount,
        int totalArtifacts,
        int pendingArtifacts,
        int blockedArtifacts,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
