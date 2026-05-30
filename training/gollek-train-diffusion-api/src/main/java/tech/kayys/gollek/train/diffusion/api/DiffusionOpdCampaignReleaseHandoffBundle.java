package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed downstream handoff bundle for final campaign release state.
 */
public record DiffusionOpdCampaignReleaseHandoffBundle(
        DiffusionOpdCampaignReleaseReceipt releaseReceipt,
        DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
        DiffusionOpdCampaignReleaseResolution releaseResolution,
        DiffusionOpdCampaignReleaseClosure releaseClosure,
        String handoffTarget,
        String exportStatus,
        boolean terminal,
        int runnableStepCount,
        int blockedEntryCount,
        String handoffKey,
        String summaryMessage) {
}
