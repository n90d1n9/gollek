package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed dispatch plan for the dispatchable subset of a campaign.
 */
public record DiffusionOpdCampaignDispatchPlan(
        DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary,
        List<DiffusionOpdCampaignDispatchPlanStep> steps,
        int stepCount,
        int blockedCount,
        int escalationRequiredCount,
        boolean executable,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
