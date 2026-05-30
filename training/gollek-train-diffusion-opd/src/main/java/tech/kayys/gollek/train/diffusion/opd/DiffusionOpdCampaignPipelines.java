package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilitySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignEngineHandoffEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionReview;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleasePacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Internal staged assembly for the main campaign orchestration flow.
 *
 * <p>These builders keep the cross-run campaign lane readable without pushing that assembly detail
 * back into {@link DiffusionOpdReportQueries}. Shared pipeline carrier records live in
 * {@link DiffusionOpdCampaignPipelineTypes}, while narrower helpers own the approval, delivery,
 * execution, and release side lanes.
 */
final class DiffusionOpdCampaignPipelines {
    private DiffusionOpdCampaignPipelines() {
    }

    static CampaignApprovalPipeline buildCampaignApprovalPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignApprovalIncident.buildCampaignApprovalPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
    }

    static CampaignIncidentHandoffPipeline buildCampaignIncidentHandoffPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdCampaignApprovalIncident.buildCampaignIncidentHandoffPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
    }

    static CampaignDeliveryReceiptPipeline buildCampaignDeliveryReceiptPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        DiffusionOpdCampaignExportManifest exportManifest = DiffusionOpdReportQueries.campaignExportManifest(
                buildCampaignIncidentHandoffPipeline(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit).handoffBundle());
        DiffusionOpdCampaignDeliveryReport deliveryReport = DiffusionOpdCampaignDeliveryPlanning
                .campaignDeliveryReport(exportManifest);
        DiffusionOpdCampaignDeliveryLedger deliveryLedger = DiffusionOpdCampaignDeliveryPlanning
                .campaignDeliveryLedger(deliveryReport);
        DiffusionOpdCampaignDeliveryReceipts deliveryReceipts = DiffusionOpdCampaignDeliveryPlanning
                .campaignDeliveryReceipts(deliveryLedger);
        return new CampaignDeliveryReceiptPipeline(
                exportManifest,
                deliveryReport,
                deliveryLedger,
                deliveryReceipts);
    }

    static CampaignAcknowledgementPlanningPipeline buildCampaignAcknowledgementPlanningPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        DiffusionOpdCampaignReceiptAcknowledgements receiptAcknowledgements =
                DiffusionOpdCampaignDeliveryPlanning.campaignReceiptAcknowledgements(buildCampaignDeliveryReceiptPipeline(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit).deliveryReceipts());
        DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions = DiffusionOpdCampaignDeliveryPlanning
                .campaignAcknowledgementDecisions(receiptAcknowledgements);
        DiffusionOpdCampaignDecisionResolutions decisionResolutions = DiffusionOpdCampaignDeliveryPlanning
                .campaignDecisionResolutions(acknowledgementDecisions);
        return new CampaignAcknowledgementPlanningPipeline(
                receiptAcknowledgements,
                acknowledgementDecisions,
                decisionResolutions);
    }

    static CampaignDispatchPlanningPipeline buildCampaignDispatchPlanningPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        CampaignAcknowledgementPlanningPipeline acknowledgementPlanningPipeline =
                buildCampaignAcknowledgementPlanningPipeline(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit);
        DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions =
                acknowledgementPlanningPipeline.acknowledgementDecisions();
        DiffusionOpdCampaignDecisionResolutions decisionResolutions =
                acknowledgementPlanningPipeline.decisionResolutions();
        DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary = DiffusionOpdCampaignDeliveryPlanning
                .campaignDispatchEligibilitySummary(decisionResolutions);
        DiffusionOpdCampaignDispatchPlan dispatchPlan = DiffusionOpdCampaignDeliveryPlanning
                .campaignDispatchPlan(eligibilitySummary);
        return new CampaignDispatchPlanningPipeline(
                acknowledgementDecisions,
                decisionResolutions,
                eligibilitySummary,
                dispatchPlan);
    }

    static CampaignDispatchExecutionPipeline buildCampaignDispatchExecutionPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        CampaignDispatchPlanningPipeline dispatchPlanningPipeline = buildCampaignDispatchPlanningPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
        DiffusionOpdCampaignExecutionPacket executionPacket = DiffusionOpdCampaignExecutionReleases
                .campaignExecutionPacket(dispatchPlanningPipeline.dispatchPlan());
        DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope = DiffusionOpdCampaignExecutionReleases
                .campaignEngineHandoffEnvelope(executionPacket);
        return new CampaignDispatchExecutionPipeline(
                dispatchPlanningPipeline.eligibilitySummary(),
                dispatchPlanningPipeline.dispatchPlan(),
                executionPacket,
                handoffEnvelope);
    }

    static CampaignExecutionReleasePipeline buildCampaignExecutionReleasePipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        CampaignDispatchExecutionPipeline dispatchExecutionPipeline = buildCampaignDispatchExecutionPipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
        DiffusionOpdCampaignExecutionManifest executionManifest = DiffusionOpdCampaignExecutionReleases
                .campaignExecutionManifest(dispatchExecutionPipeline.handoffEnvelope());
        DiffusionOpdCampaignExecutionReview executionReview = DiffusionOpdCampaignExecutionReleases
                .campaignExecutionReview(executionManifest);
        DiffusionOpdCampaignReleaseDecision releaseDecision = DiffusionOpdCampaignExecutionReleases
                .campaignReleaseDecision(executionReview);
        return new CampaignExecutionReleasePipeline(
                dispatchExecutionPipeline.handoffEnvelope(),
                executionManifest,
                executionReview,
                releaseDecision);
    }

    static CampaignReleasePipeline buildCampaignReleasePipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        CampaignExecutionReleasePipeline executionReleasePipeline = buildCampaignExecutionReleasePipeline(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
        DiffusionOpdCampaignReleaseDecision releaseDecision = executionReleasePipeline.releaseDecision();
        DiffusionOpdCampaignReleasePacket releasePacket = DiffusionOpdCampaignReleaseStages
                .campaignReleasePacket(releaseDecision);
        DiffusionOpdCampaignReleaseLedger releaseLedger = DiffusionOpdCampaignReleaseStages
                .campaignReleaseLedger(releasePacket);
        DiffusionOpdCampaignReleaseReceipt releaseReceipt = DiffusionOpdCampaignReleaseStages
                .campaignReleaseReceipt(releaseLedger);
        DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement = DiffusionOpdCampaignReleaseStages
                .campaignReleaseAcknowledgement(releaseReceipt);
        DiffusionOpdCampaignReleaseResolution releaseResolution = DiffusionOpdCampaignReleaseStages
                .campaignReleaseResolution(releaseAcknowledgement);
        DiffusionOpdCampaignReleaseClosure releaseClosure = DiffusionOpdCampaignReleaseStages
                .campaignReleaseClosure(releaseResolution);
        DiffusionOpdCampaignReleaseHandoffBundle handoffBundle = DiffusionOpdCampaignReleaseStages
                .campaignReleaseHandoffBundle(
                        releaseReceipt,
                        releaseAcknowledgement,
                        releaseResolution,
                        releaseClosure);
        DiffusionOpdCampaignReleaseExportManifest exportManifest = DiffusionOpdCampaignReleaseStages
                .campaignReleaseExportManifest(handoffBundle);
        return new CampaignReleasePipeline(
                releasePacket,
                releaseLedger,
                releaseReceipt,
                releaseAcknowledgement,
                releaseResolution,
                releaseClosure,
                handoffBundle,
                exportManifest);
    }
}
