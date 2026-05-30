package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed operator-facing execution review for a campaign manifest.
 */
public record DiffusionOpdCampaignExecutionReview(
        DiffusionOpdCampaignExecutionManifest executionManifest,
        String reviewerTarget,
        String reviewOutcome,
        boolean approvalRequired,
        boolean releaseReady,
        int runnableStepCount,
        int blockedEntryCount,
        String reviewKey,
        String summaryMessage) {
}
