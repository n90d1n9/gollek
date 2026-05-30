package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignApprovalPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignClosureReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionLog;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignIncidentReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignResolutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignRolloutPack;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignSummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Owns the approval and incident-handoff campaign lane so broader campaign
 * pipeline assembly can stay focused on later downstream stages.
 *
 * <p>This is the earliest campaign lane: it turns workflow batches into approval state, then into
 * the incident/handoff bundle that downstream delivery planning consumes.
 */
final class DiffusionOpdCampaignApprovalIncident {
    private DiffusionOpdCampaignApprovalIncident() {
    }

    static CampaignApprovalPipeline buildCampaignApprovalPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        DiffusionOpdCampaignSummary campaignSummary = DiffusionOpdReportQueries.campaignSummary(
                DiffusionOpdReportQueries.taskPortfolioWorkflowBatchManifest(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit),
                DiffusionOpdReportQueries.taskStagePortfolioWorkflowBatchManifest(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit));
        DiffusionOpdCampaignRolloutPack rolloutPack = DiffusionOpdReportQueries.campaignRolloutPack(campaignSummary);
        DiffusionOpdCampaignApprovalPacket approvalPacket =
                DiffusionOpdReportQueries.campaignApprovalPacket(rolloutPack);
        return new CampaignApprovalPipeline(
                campaignSummary,
                rolloutPack,
                approvalPacket);
    }

    static CampaignIncidentHandoffPipeline buildCampaignIncidentHandoffPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        DiffusionOpdCampaignDecisionLog decisionLog = DiffusionOpdReportQueries.campaignDecisionLog(
                buildCampaignApprovalPipeline(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit).approvalPacket());
        DiffusionOpdCampaignIncidentReport incidentReport = DiffusionOpdReportQueries.campaignIncidentReport(decisionLog);
        DiffusionOpdCampaignResolutionPacket resolutionPacket =
                DiffusionOpdReportQueries.campaignResolutionPacket(incidentReport);
        DiffusionOpdCampaignClosureReport closureReport =
                DiffusionOpdReportQueries.campaignClosureReport(resolutionPacket);
        DiffusionOpdCampaignHandoffBundle handoffBundle = DiffusionOpdReportQueries.campaignHandoffBundle(
                decisionLog,
                incidentReport,
                resolutionPacket,
                closureReport);
        return new CampaignIncidentHandoffPipeline(
                decisionLog,
                incidentReport,
                resolutionPacket,
                closureReport,
                handoffBundle);
    }
}
