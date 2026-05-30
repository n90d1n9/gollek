package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign execution step for one ordered batch.
 */
public record DiffusionOpdCampaignExecutionStep(
        String batchScope,
        String batchKey,
        String dispatchPriority,
        int targetSliceCount,
        int approvalRequiredCount,
        Integer recheckAfterRounds) {
}
