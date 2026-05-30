package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed acknowledgement outcome for a campaign release receipt.
 */
public record DiffusionOpdCampaignReleaseAcknowledgement(
        DiffusionOpdCampaignReleaseReceipt releaseReceipt,
        String acknowledgementOutcome,
        String reviewer,
        String reviewerRole,
        boolean finalAcceptance,
        boolean followUpRequired,
        int runnableStepCount,
        int blockedEntryCount,
        String acknowledgementKey,
        String summaryMessage) {
}
