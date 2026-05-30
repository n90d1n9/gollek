package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed delivery decision collection for release delivery acknowledgements.
 */
public record DiffusionOpdCampaignReleaseDeliveryDecisions(
        DiffusionOpdCampaignReleaseDeliveryAcknowledgements deliveryAcknowledgements,
        List<DiffusionOpdCampaignReleaseDeliveryDecision> decisions,
        int decisionCount,
        int acceptedCount,
        int deferredCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
