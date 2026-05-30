package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed release delivery report for downstream integrations.
 */
public record DiffusionOpdCampaignReleaseDeliveryReport(
        DiffusionOpdCampaignReleaseExportManifest exportManifest,
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
