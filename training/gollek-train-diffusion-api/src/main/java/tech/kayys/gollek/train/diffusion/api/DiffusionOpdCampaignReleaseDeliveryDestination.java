package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed destination-level release delivery summary.
 */
public record DiffusionOpdCampaignReleaseDeliveryDestination(
        String destination,
        List<String> artifactTypes,
        int artifactCount,
        int pendingCount,
        int blockedCount,
        boolean followUpRequired,
        String summaryMessage) {
}
