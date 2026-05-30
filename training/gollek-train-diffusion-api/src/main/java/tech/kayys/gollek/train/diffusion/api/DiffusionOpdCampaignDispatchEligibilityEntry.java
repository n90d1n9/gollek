package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed dispatch eligibility entry for one resolved campaign slice.
 */
public record DiffusionOpdCampaignDispatchEligibilityEntry(
        String receiptId,
        String artifactType,
        String eligibility,
        String executionReadiness,
        String escalationOwner,
        boolean dispatchable,
        String summaryMessage) {
}
