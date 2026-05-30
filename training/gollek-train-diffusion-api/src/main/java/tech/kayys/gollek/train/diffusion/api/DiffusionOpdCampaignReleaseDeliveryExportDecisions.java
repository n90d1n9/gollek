package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed delivery decision collection for release-delivery export acknowledgements.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportDecisions(
        DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements,
        List<DiffusionOpdCampaignReleaseDeliveryExportDecision> decisions,
        int decisionCount,
        int acceptedCount,
        int deferredCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
