package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign delivery summary for one downstream destination.
 */
public record DiffusionOpdCampaignDeliveryDestination(
        String destination,
        List<String> artifactTypes,
        int artifactCount,
        int deliveredCount,
        int pendingCount,
        int blockedCount,
        boolean followUpRequired,
        String summaryMessage) {
}
