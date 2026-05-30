package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed operator-facing workflow campaign incident report.
 */
public record DiffusionOpdCampaignIncidentReport(
        DiffusionOpdCampaignDecisionLog decisionLog,
        Boolean incidentOpen,
        String highestSeverity,
        int blockedBatchCount,
        List<String> blockedBatchScopes,
        List<String> approvalReasonCodes,
        String primaryOperatorTarget,
        String recommendedOperatorAction,
        String summaryMessage) {
}
