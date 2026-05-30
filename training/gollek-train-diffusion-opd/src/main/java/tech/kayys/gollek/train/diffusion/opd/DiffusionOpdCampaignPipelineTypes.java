package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignApprovalPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignClosureReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionLog;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignIncidentReport;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignResolutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignRolloutPack;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignSummary;

/**
 * Shared package-private carrier records for staged campaign assembly.
 *
 * <p>These records intentionally stay logic-free so builder helpers can share intermediate shapes
 * without re-nesting them inside one specific orchestration class. They are ordered from the
 * earliest approval/incident lane through delivery, dispatch, execution, and final release state.
 */
record CampaignApprovalPipeline(
        DiffusionOpdCampaignSummary campaignSummary,
        DiffusionOpdCampaignRolloutPack rolloutPack,
        DiffusionOpdCampaignApprovalPacket approvalPacket) {
}

record CampaignIncidentHandoffPipeline(
        DiffusionOpdCampaignDecisionLog decisionLog,
        DiffusionOpdCampaignIncidentReport incidentReport,
        DiffusionOpdCampaignResolutionPacket resolutionPacket,
        DiffusionOpdCampaignClosureReport closureReport,
        DiffusionOpdCampaignHandoffBundle handoffBundle) {
}

record CampaignDeliveryReceiptPipeline(
        DiffusionOpdCampaignExportManifest exportManifest,
        DiffusionOpdCampaignDeliveryReport deliveryReport,
        DiffusionOpdCampaignDeliveryLedger deliveryLedger,
        DiffusionOpdCampaignDeliveryReceipts deliveryReceipts) {
}

record CampaignAcknowledgementPlanningPipeline(
        DiffusionOpdCampaignReceiptAcknowledgements receiptAcknowledgements,
        DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions,
        DiffusionOpdCampaignDecisionResolutions decisionResolutions) {
}

record CampaignDispatchPlanningPipeline(
        DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions,
        DiffusionOpdCampaignDecisionResolutions decisionResolutions,
        DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary,
        DiffusionOpdCampaignDispatchPlan dispatchPlan) {
}

record CampaignDispatchExecutionPipeline(
        DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary,
        DiffusionOpdCampaignDispatchPlan dispatchPlan,
        DiffusionOpdCampaignExecutionPacket executionPacket,
        DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope) {
}

record CampaignExecutionReleasePipeline(
        DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope,
        DiffusionOpdCampaignExecutionManifest executionManifest,
        DiffusionOpdCampaignExecutionReview executionReview,
        DiffusionOpdCampaignReleaseDecision releaseDecision) {
}

record CampaignReleasePipeline(
        DiffusionOpdCampaignReleasePacket releasePacket,
        DiffusionOpdCampaignReleaseLedger releaseLedger,
        DiffusionOpdCampaignReleaseReceipt releaseReceipt,
        DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
        DiffusionOpdCampaignReleaseResolution releaseResolution,
        DiffusionOpdCampaignReleaseClosure releaseClosure,
        DiffusionOpdCampaignReleaseHandoffBundle handoffBundle,
        DiffusionOpdCampaignReleaseExportManifest exportManifest) {
}
