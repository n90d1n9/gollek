package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign decision log for audit and review trails.
 */
public record DiffusionOpdCampaignDecisionLog(
        DiffusionOpdCampaignApprovalPacket approvalPacket,
        List<DiffusionOpdCampaignDecisionLogEntry> entries,
        int decisionCount,
        int gatedDecisionCount,
        String highestSeverity,
        String summaryMessage) {
}
