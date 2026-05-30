package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign approval step for one gated batch.
 */
public record DiffusionOpdCampaignApprovalStep(
        String batchScope,
        String batchKey,
        Boolean signOffRequired,
        String reasonCode,
        int targetSliceCount,
        String approverTarget) {
}
