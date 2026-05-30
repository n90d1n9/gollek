package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed final resolution collection for campaign acknowledgement decisions.
 */
public record DiffusionOpdCampaignDecisionResolutions(
        DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions,
        List<DiffusionOpdCampaignDecisionResolution> resolutions,
        int resolutionCount,
        int executableCount,
        int blockedCount,
        int escalationRequiredCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
