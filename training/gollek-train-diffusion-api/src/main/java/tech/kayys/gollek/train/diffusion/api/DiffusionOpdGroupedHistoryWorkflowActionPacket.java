package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed grouped workflow-ready action packet derived from a policy rollout plan.
 */
public record DiffusionOpdGroupedHistoryWorkflowActionPacket(
        DiffusionOpdGroupedHistoryPortfolioPolicyRolloutPlan rolloutPlan,
        String selectedPolicyName,
        String fallbackPolicyName,
        String primaryAction,
        String fallbackAction,
        Boolean automationFriendly,
        Boolean reviewRequired,
        int reviewHotspotCount,
        int manualReviewCount,
        String summaryMessage) {
}
