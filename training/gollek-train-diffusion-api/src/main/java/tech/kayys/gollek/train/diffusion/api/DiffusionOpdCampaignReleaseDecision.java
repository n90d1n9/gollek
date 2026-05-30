package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed release decision for a reviewed campaign execution.
 */
public record DiffusionOpdCampaignReleaseDecision(
        DiffusionOpdCampaignExecutionReview executionReview,
        String releaseOutcome,
        String fallbackOutcome,
        boolean dispatchAllowed,
        boolean operatorFollowUpRequired,
        int runnableStepCount,
        int blockedEntryCount,
        String decisionKey,
        String summaryMessage) {
}
