package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed workflow campaign handoff bundle for downstream schedulers and review tools.
 */
public record DiffusionOpdCampaignHandoffBundle(
        DiffusionOpdCampaignDecisionLog decisionLog,
        DiffusionOpdCampaignIncidentReport incidentReport,
        DiffusionOpdCampaignResolutionPacket resolutionPacket,
        DiffusionOpdCampaignClosureReport closureReport,
        String handoffTarget,
        String exportStatus,
        String summaryMessage) {
}
