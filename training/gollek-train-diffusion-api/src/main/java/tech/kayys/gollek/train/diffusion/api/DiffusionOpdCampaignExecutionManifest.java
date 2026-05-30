package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed engine-ready execution manifest for a campaign handoff.
 */
public record DiffusionOpdCampaignExecutionManifest(
        DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope,
        String queueName,
        String batchMode,
        String retryWindow,
        int queueOrder,
        boolean reviewRequired,
        boolean dispatchReady,
        int runnableStepCount,
        int blockedEntryCount,
        String manifestKey,
        String summaryMessage) {
}
