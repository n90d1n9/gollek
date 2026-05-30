package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign delivery report for downstream integration tracking.
 */
public record DiffusionOpdCampaignDeliveryReport(
        DiffusionOpdCampaignExportManifest exportManifest,
        List<DiffusionOpdCampaignDeliveryDestination> destinations,
        int destinationCount,
        int totalArtifacts,
        int pendingArtifacts,
        int blockedArtifacts,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
