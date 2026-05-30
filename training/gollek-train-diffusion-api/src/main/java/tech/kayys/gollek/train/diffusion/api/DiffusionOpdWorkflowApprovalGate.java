package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow approval gate for scheduler-ready execution planning.
 */
public record DiffusionOpdWorkflowApprovalGate(
        Boolean approvalRequired,
        String gateCode,
        String escalationTarget,
        Boolean reviewRequired,
        int manualReviewCount,
        int reviewHotspotCount) {
}
