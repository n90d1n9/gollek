package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign closure report for final auditable outcome packaging.
 */
public record DiffusionOpdCampaignClosureReport(
        DiffusionOpdCampaignResolutionPacket resolutionPacket,
        String finalOutcome,
        Boolean closed,
        int unresolvedOverrideCount,
        String followUpAction,
        String summaryMessage) {
}
