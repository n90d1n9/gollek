package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed terminal closure state for one release delivery resolution.
 */
public record DiffusionOpdCampaignReleaseDeliveryClosure(
        String receiptId,
        String artifactType,
        String destination,
        String finalOutcome,
        boolean closed,
        String followUpAction,
        boolean dispatchComplete,
        String summaryMessage) {
}
