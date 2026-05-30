package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed final resolution for a campaign release acknowledgement.
 */
public record DiffusionOpdCampaignReleaseResolution(
        DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
        String resolution,
        String escalationTarget,
        boolean finalAccepted,
        boolean dispatchEligible,
        int runnableStepCount,
        int blockedEntryCount,
        String resolutionKey,
        String summaryMessage) {
}
