package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign rollout pack for scheduler or UI orchestration.
 */
public record DiffusionOpdCampaignRolloutPack(
        DiffusionOpdCampaignSummary campaignSummary,
        String recommendedExecutionMode,
        String fallbackExecutionMode,
        String primaryStrategy,
        Boolean approvalRequired,
        List<DiffusionOpdCampaignExecutionStep> batchExecutionSteps,
        String summaryMessage) {
}
