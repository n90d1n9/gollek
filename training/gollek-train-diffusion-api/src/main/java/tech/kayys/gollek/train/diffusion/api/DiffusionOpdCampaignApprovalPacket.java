package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign approval packet for human sign-off flows.
 */
public record DiffusionOpdCampaignApprovalPacket(
        DiffusionOpdCampaignRolloutPack rolloutPack,
        Boolean signOffRequired,
        String primaryApproverTarget,
        List<String> approvalReasonCodes,
        List<DiffusionOpdCampaignApprovalStep> gatedBatchSteps,
        int blockedBatchCount,
        String fallbackExecutionMode,
        String summaryMessage) {
}
