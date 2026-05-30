package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed engine-facing handoff envelope for a campaign execution packet.
 */
public record DiffusionOpdCampaignEngineHandoffEnvelope(
        DiffusionOpdCampaignExecutionPacket executionPacket,
        String engineId,
        String submissionMode,
        String retryPolicy,
        boolean requiresHumanReview,
        boolean readyForDispatch,
        int runnableStepCount,
        int blockedEntryCount,
        String handoffKey,
        String summaryMessage) {
}
