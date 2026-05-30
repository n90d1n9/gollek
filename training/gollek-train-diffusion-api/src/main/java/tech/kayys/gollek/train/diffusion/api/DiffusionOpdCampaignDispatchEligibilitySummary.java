package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed dispatch eligibility summary for campaign decision resolutions.
 */
public record DiffusionOpdCampaignDispatchEligibilitySummary(
        DiffusionOpdCampaignDecisionResolutions decisionResolutions,
        List<DiffusionOpdCampaignDispatchEligibilityEntry> entries,
        int entryCount,
        int dispatchableCount,
        int blockedCount,
        int escalationRequiredCount,
        boolean anyDispatchable,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
