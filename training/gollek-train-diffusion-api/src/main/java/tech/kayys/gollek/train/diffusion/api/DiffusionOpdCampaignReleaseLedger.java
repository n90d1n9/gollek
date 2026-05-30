package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed audit ledger for final campaign release handoff state.
 */
public record DiffusionOpdCampaignReleaseLedger(
        DiffusionOpdCampaignReleasePacket releasePacket,
        String releaseStatus,
        String acknowledgementStatus,
        int attemptCount,
        boolean handoffComplete,
        boolean followUpRequired,
        int runnableStepCount,
        int blockedEntryCount,
        String ledgerKey,
        String summaryMessage) {
}
