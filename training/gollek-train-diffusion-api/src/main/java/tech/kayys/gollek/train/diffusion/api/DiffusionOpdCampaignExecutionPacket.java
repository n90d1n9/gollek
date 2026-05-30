package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed engine-facing execution packet for a campaign dispatch plan.
 */
public record DiffusionOpdCampaignExecutionPacket(
        DiffusionOpdCampaignDispatchPlan dispatchPlan,
        List<DiffusionOpdCampaignDispatchPlanStep> runnableSteps,
        List<DiffusionOpdCampaignDispatchEligibilityEntry> blockedEntries,
        int runnableStepCount,
        int blockedEntryCount,
        int escalationRequiredCount,
        boolean executable,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
