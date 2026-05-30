package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign resolution step for one batch override recommendation.
 */
public record DiffusionOpdCampaignResolutionStep(
        String batchScope,
        String batchKey,
        String recommendedOutcome,
        Boolean overrideAllowed,
        String reasonCode) {
}
