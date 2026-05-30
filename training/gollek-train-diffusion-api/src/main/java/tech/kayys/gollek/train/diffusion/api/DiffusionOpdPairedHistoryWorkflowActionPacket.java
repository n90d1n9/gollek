package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed paired workflow-ready action packet derived from a policy rollout plan.
 */
public record DiffusionOpdPairedHistoryWorkflowActionPacket(
        DiffusionOpdPairedHistoryPortfolioPolicyRolloutPlan rolloutPlan,
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
