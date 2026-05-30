package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed terminal closure state for a campaign release resolution.
 */
public record DiffusionOpdCampaignReleaseClosure(
        DiffusionOpdCampaignReleaseResolution releaseResolution,
        String finalOutcome,
        boolean closed,
        String followUpAction,
        boolean dispatchComplete,
        int runnableStepCount,
        int blockedEntryCount,
        String closureKey,
        String summaryMessage) {
}
