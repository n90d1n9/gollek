package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilityEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlanStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignEngineHandoffEnvelope;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionPacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExecutionReview;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleasePacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseResolution;

/**
 * Internal execution, review, and release-stage assembly for campaign workflows.
 *
 * <p>This keeps the execution-to-release lane together so the public query class can stay focused
 * on entrypoints rather than step-by-step release orchestration. Final release-terminal shaping is
 * intentionally delegated to {@link DiffusionOpdCampaignReleaseStages}.
 */
final class DiffusionOpdCampaignExecutionReleases {
    private DiffusionOpdCampaignExecutionReleases() {
    }

    static DiffusionOpdCampaignExecutionPacket campaignExecutionPacket(
            DiffusionOpdCampaignDispatchPlan dispatchPlan) {
        List<DiffusionOpdCampaignDispatchPlanStep> runnableSteps = dispatchPlan.steps();
        List<DiffusionOpdCampaignDispatchEligibilityEntry> blockedEntries = dispatchPlan.eligibilitySummary().entries().stream()
                .filter(entry -> !entry.dispatchable())
                .toList();
        return new DiffusionOpdCampaignExecutionPacket(
                dispatchPlan,
                runnableSteps,
                blockedEntries,
                runnableSteps.size(),
                blockedEntries.size(),
                dispatchPlan.escalationRequiredCount(),
                dispatchPlan.executable(),
                dispatchPlan.primaryDestination(),
                dispatchPlan.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignExecutionPacketMessage(
                        runnableSteps.size(),
                        blockedEntries.size(),
                        dispatchPlan.escalationRequiredCount()));
    }

    static DiffusionOpdCampaignEngineHandoffEnvelope campaignEngineHandoffEnvelope(
            DiffusionOpdCampaignExecutionPacket executionPacket) {
        boolean readyForDispatch = executionPacket.executable();
        boolean requiresHumanReview = !readyForDispatch;
        String submissionMode = readyForDispatch ? "direct_dispatch" : "review_queue";
        String retryPolicy = readyForDispatch ? "standard_retry" : "await_review";
        String engineId = executionPacket.primaryDestination() == null
                ? "campaign-engine"
                : executionPacket.primaryDestination();
        String handoffKey = engineId
                + ":"
                + submissionMode
                + ":"
                + executionPacket.runnableStepCount()
                + ":"
                + executionPacket.blockedEntryCount();
        return new DiffusionOpdCampaignEngineHandoffEnvelope(
                executionPacket,
                engineId,
                submissionMode,
                retryPolicy,
                requiresHumanReview,
                readyForDispatch,
                executionPacket.runnableStepCount(),
                executionPacket.blockedEntryCount(),
                handoffKey,
                DiffusionOpdReportQueries.buildCampaignEngineHandoffEnvelopeMessage(
                        engineId,
                        submissionMode,
                        executionPacket.runnableStepCount(),
                        executionPacket.blockedEntryCount()));
    }

    static DiffusionOpdCampaignExecutionManifest campaignExecutionManifest(
            DiffusionOpdCampaignEngineHandoffEnvelope handoffEnvelope) {
        boolean reviewRequired = handoffEnvelope.requiresHumanReview();
        boolean dispatchReady = handoffEnvelope.readyForDispatch();
        String queueName = reviewRequired ? "campaign-review" : "campaign-dispatch";
        String batchMode = handoffEnvelope.runnableStepCount() > 1 ? "parallel" : "serial";
        String retryWindow = reviewRequired ? "after_review" : "standard_window";
        int queueOrder = reviewRequired ? 100 : 10;
        String manifestKey = handoffEnvelope.engineId()
                + ":"
                + queueName
                + ":"
                + handoffEnvelope.runnableStepCount()
                + ":"
                + handoffEnvelope.blockedEntryCount();
        return new DiffusionOpdCampaignExecutionManifest(
                handoffEnvelope,
                queueName,
                batchMode,
                retryWindow,
                queueOrder,
                reviewRequired,
                dispatchReady,
                handoffEnvelope.runnableStepCount(),
                handoffEnvelope.blockedEntryCount(),
                manifestKey,
                DiffusionOpdReportQueries.buildCampaignExecutionManifestMessage(
                        queueName,
                        batchMode,
                        handoffEnvelope.runnableStepCount(),
                        handoffEnvelope.blockedEntryCount()));
    }

    static DiffusionOpdCampaignExecutionReview campaignExecutionReview(
            DiffusionOpdCampaignExecutionManifest executionManifest) {
        boolean approvalRequired = executionManifest.reviewRequired();
        boolean releaseReady = executionManifest.dispatchReady();
        String reviewerTarget = approvalRequired ? "training-monitor" : "dispatch-engine";
        String reviewOutcome = releaseReady ? "approved_for_release" : "pending_review";
        String reviewKey = executionManifest.queueName()
                + ":"
                + reviewerTarget
                + ":"
                + executionManifest.runnableStepCount()
                + ":"
                + executionManifest.blockedEntryCount();
        return new DiffusionOpdCampaignExecutionReview(
                executionManifest,
                reviewerTarget,
                reviewOutcome,
                approvalRequired,
                releaseReady,
                executionManifest.runnableStepCount(),
                executionManifest.blockedEntryCount(),
                reviewKey,
                DiffusionOpdReportQueries.buildCampaignExecutionReviewMessage(
                        reviewerTarget,
                        reviewOutcome,
                        executionManifest.runnableStepCount(),
                        executionManifest.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseDecision campaignReleaseDecision(
            DiffusionOpdCampaignExecutionReview executionReview) {
        boolean dispatchAllowed = executionReview.releaseReady();
        boolean operatorFollowUpRequired = !dispatchAllowed;
        String releaseOutcome = dispatchAllowed ? "approved" : "deferred";
        String fallbackOutcome = dispatchAllowed ? "deferred" : "blocked";
        String decisionKey = executionReview.reviewerTarget()
                + ":"
                + releaseOutcome
                + ":"
                + executionReview.runnableStepCount()
                + ":"
                + executionReview.blockedEntryCount();
        return new DiffusionOpdCampaignReleaseDecision(
                executionReview,
                releaseOutcome,
                fallbackOutcome,
                dispatchAllowed,
                operatorFollowUpRequired,
                executionReview.runnableStepCount(),
                executionReview.blockedEntryCount(),
                decisionKey,
                DiffusionOpdReportQueries.buildCampaignReleaseDecisionMessage(
                        releaseOutcome,
                        fallbackOutcome,
                        executionReview.runnableStepCount(),
                        executionReview.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleasePacket campaignReleasePacket(
            DiffusionOpdCampaignReleaseDecision releaseDecision) {
        return DiffusionOpdCampaignReleaseStages.campaignReleasePacket(releaseDecision);
    }

    static DiffusionOpdCampaignReleaseLedger campaignReleaseLedger(
            DiffusionOpdCampaignReleasePacket releasePacket) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseLedger(releasePacket);
    }

    static DiffusionOpdCampaignReleaseReceipt campaignReleaseReceipt(
            DiffusionOpdCampaignReleaseLedger releaseLedger) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseReceipt(releaseLedger);
    }

    static DiffusionOpdCampaignReleaseAcknowledgement campaignReleaseAcknowledgement(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseAcknowledgement(releaseReceipt);
    }

    static DiffusionOpdCampaignReleaseResolution campaignReleaseResolution(
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseResolution(releaseAcknowledgement);
    }

    static DiffusionOpdCampaignReleaseClosure campaignReleaseClosure(
            DiffusionOpdCampaignReleaseResolution releaseResolution) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseClosure(releaseResolution);
    }

    static DiffusionOpdCampaignReleaseHandoffBundle campaignReleaseHandoffBundle(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt,
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
            DiffusionOpdCampaignReleaseResolution releaseResolution,
            DiffusionOpdCampaignReleaseClosure releaseClosure) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseHandoffBundle(
                releaseReceipt,
                releaseAcknowledgement,
                releaseResolution,
                releaseClosure);
    }

    static DiffusionOpdCampaignReleaseExportManifest campaignReleaseExportManifest(
            DiffusionOpdCampaignReleaseHandoffBundle handoffBundle) {
        return DiffusionOpdCampaignReleaseStages.campaignReleaseExportManifest(handoffBundle);
    }
}
