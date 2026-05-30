package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed confirmation receipt for a campaign release handoff.
 */
public record DiffusionOpdCampaignReleaseReceipt(
        DiffusionOpdCampaignReleaseLedger releaseLedger,
        String receiptId,
        String receiptStatus,
        String finalAcknowledgement,
        boolean confirmed,
        boolean followUpRequired,
        int runnableStepCount,
        int blockedEntryCount,
        String receiptKey,
        String summaryMessage) {
}
