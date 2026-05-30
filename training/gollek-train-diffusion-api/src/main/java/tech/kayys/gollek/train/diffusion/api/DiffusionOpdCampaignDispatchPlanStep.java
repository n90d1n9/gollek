package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed dispatch plan step for one dispatchable campaign slice.
 */
public record DiffusionOpdCampaignDispatchPlanStep(
        int order,
        String receiptId,
        String artifactType,
        String executionMode,
        String dispatchTarget,
        String summaryMessage) {
}
