package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed workflow campaign resolution packet for operator response flows.
 */
public record DiffusionOpdCampaignResolutionPacket(
        DiffusionOpdCampaignIncidentReport incidentReport,
        String suggestedOutcome,
        String fallbackOutcome,
        Boolean operatorResponseRequired,
        List<DiffusionOpdCampaignResolutionStep> batchResolutions,
        String summaryMessage) {
}
