package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed terminal closure state for one release-delivery export resolution.
 */
public record DiffusionOpdCampaignReleaseDeliveryExportClosure(
        String receiptId,
        String artifactType,
        String destination,
        String finalOutcome,
        boolean closed,
        String followUpAction,
        boolean dispatchComplete,
        String summaryMessage) {
}
