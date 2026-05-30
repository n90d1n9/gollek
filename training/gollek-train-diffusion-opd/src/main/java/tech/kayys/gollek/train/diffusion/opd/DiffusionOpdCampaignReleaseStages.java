package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleasePacket;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseResolution;

/**
 * Owns the release-terminal stage projections so
 * {@link DiffusionOpdCampaignExecutionReleases} can stop at release decision assembly while this
 * helper handles final handoff and export shaping.
 */
final class DiffusionOpdCampaignReleaseStages {

    private DiffusionOpdCampaignReleaseStages() {
    }

    static DiffusionOpdCampaignReleasePacket campaignReleasePacket(
            DiffusionOpdCampaignReleaseDecision releaseDecision) {
        boolean readyForEngineHandoff = releaseDecision.dispatchAllowed();
        boolean reviewEscalationOpen = releaseDecision.operatorFollowUpRequired();
        String dispatchTarget = readyForEngineHandoff ? "dispatch-engine" : "training-monitor";
        String packetOutcome = readyForEngineHandoff ? "handoff_ready" : "review_handoff";
        String fallbackPacketOutcome = readyForEngineHandoff ? "review_handoff" : "blocked_handoff";
        String packetKey = dispatchTarget
                + ":"
                + packetOutcome
                + ":"
                + releaseDecision.runnableStepCount()
                + ":"
                + releaseDecision.blockedEntryCount();
        return new DiffusionOpdCampaignReleasePacket(
                releaseDecision,
                dispatchTarget,
                packetOutcome,
                fallbackPacketOutcome,
                readyForEngineHandoff,
                reviewEscalationOpen,
                releaseDecision.runnableStepCount(),
                releaseDecision.blockedEntryCount(),
                packetKey,
                DiffusionOpdReportQueries.buildCampaignReleasePacketMessage(
                        dispatchTarget,
                        packetOutcome,
                        releaseDecision.runnableStepCount(),
                        releaseDecision.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseLedger campaignReleaseLedger(
            DiffusionOpdCampaignReleasePacket releasePacket) {
        boolean handoffComplete = releasePacket.readyForEngineHandoff();
        boolean followUpRequired = releasePacket.reviewEscalationOpen();
        String releaseStatus = handoffComplete ? "released" : "pending_review";
        String acknowledgementStatus = handoffComplete ? "accepted" : "awaiting_ack";
        int attemptCount = handoffComplete ? 1 : 0;
        String ledgerKey = releasePacket.dispatchTarget()
                + ":"
                + releaseStatus
                + ":"
                + releasePacket.runnableStepCount()
                + ":"
                + releasePacket.blockedEntryCount();
        return new DiffusionOpdCampaignReleaseLedger(
                releasePacket,
                releaseStatus,
                acknowledgementStatus,
                attemptCount,
                handoffComplete,
                followUpRequired,
                releasePacket.runnableStepCount(),
                releasePacket.blockedEntryCount(),
                ledgerKey,
                DiffusionOpdReportQueries.buildCampaignReleaseLedgerMessage(
                        releaseStatus,
                        acknowledgementStatus,
                        releasePacket.runnableStepCount(),
                        releasePacket.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseReceipt campaignReleaseReceipt(
            DiffusionOpdCampaignReleaseLedger releaseLedger) {
        boolean confirmed = releaseLedger.handoffComplete();
        boolean followUpRequired = releaseLedger.followUpRequired();
        String receiptStatus = confirmed ? "confirmed" : "pending_confirmation";
        String finalAcknowledgement = confirmed ? "accepted" : "awaiting_operator_confirmation";
        String receiptId = releaseLedger.ledgerKey() + ":receipt";
        String receiptKey = releaseLedger.releasePacket().dispatchTarget()
                + ":"
                + receiptStatus
                + ":"
                + releaseLedger.runnableStepCount()
                + ":"
                + releaseLedger.blockedEntryCount();
        return new DiffusionOpdCampaignReleaseReceipt(
                releaseLedger,
                receiptId,
                receiptStatus,
                finalAcknowledgement,
                confirmed,
                followUpRequired,
                releaseLedger.runnableStepCount(),
                releaseLedger.blockedEntryCount(),
                receiptKey,
                DiffusionOpdReportQueries.buildCampaignReleaseReceiptMessage(
                        receiptStatus,
                        finalAcknowledgement,
                        releaseLedger.runnableStepCount(),
                        releaseLedger.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseAcknowledgement campaignReleaseAcknowledgement(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt) {
        boolean finalAcceptance = releaseReceipt.confirmed();
        boolean followUpRequired = releaseReceipt.followUpRequired();
        String acknowledgementOutcome = finalAcceptance ? "accepted" : "pending_review";
        String reviewer = finalAcceptance ? "dispatch-engine" : "release-bot";
        String reviewerRole = finalAcceptance ? "engine" : "operator-gate";
        String acknowledgementKey = releaseReceipt.receiptId()
                + ":"
                + acknowledgementOutcome
                + ":"
                + releaseReceipt.runnableStepCount()
                + ":"
                + releaseReceipt.blockedEntryCount();
        return new DiffusionOpdCampaignReleaseAcknowledgement(
                releaseReceipt,
                acknowledgementOutcome,
                reviewer,
                reviewerRole,
                finalAcceptance,
                followUpRequired,
                releaseReceipt.runnableStepCount(),
                releaseReceipt.blockedEntryCount(),
                acknowledgementKey,
                DiffusionOpdReportQueries.buildCampaignReleaseAcknowledgementMessage(
                        acknowledgementOutcome,
                        reviewer,
                        releaseReceipt.runnableStepCount(),
                        releaseReceipt.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseResolution campaignReleaseResolution(
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement) {
        boolean finalAccepted = releaseAcknowledgement.finalAcceptance();
        boolean dispatchEligible = finalAccepted;
        String resolution = finalAccepted ? "accepted" : "deferred";
        String escalationTarget = finalAccepted ? "dispatch-engine" : "training-monitor";
        String resolutionKey = releaseAcknowledgement.acknowledgementKey()
                + ":"
                + resolution;
        return new DiffusionOpdCampaignReleaseResolution(
                releaseAcknowledgement,
                resolution,
                escalationTarget,
                finalAccepted,
                dispatchEligible,
                releaseAcknowledgement.runnableStepCount(),
                releaseAcknowledgement.blockedEntryCount(),
                resolutionKey,
                DiffusionOpdReportQueries.buildCampaignReleaseResolutionMessage(
                        resolution,
                        escalationTarget,
                        releaseAcknowledgement.runnableStepCount(),
                        releaseAcknowledgement.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseClosure campaignReleaseClosure(
            DiffusionOpdCampaignReleaseResolution releaseResolution) {
        boolean closed = releaseResolution.finalAccepted();
        boolean dispatchComplete = releaseResolution.dispatchEligible();
        String finalOutcome = closed ? "accepted" : "pending_follow_up";
        String followUpAction = closed ? "none" : "await_manual_review";
        String closureKey = releaseResolution.resolutionKey()
                + ":"
                + finalOutcome;
        return new DiffusionOpdCampaignReleaseClosure(
                releaseResolution,
                finalOutcome,
                closed,
                followUpAction,
                dispatchComplete,
                releaseResolution.runnableStepCount(),
                releaseResolution.blockedEntryCount(),
                closureKey,
                DiffusionOpdReportQueries.buildCampaignReleaseClosureMessage(
                        finalOutcome,
                        followUpAction,
                        releaseResolution.runnableStepCount(),
                        releaseResolution.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseHandoffBundle campaignReleaseHandoffBundle(
            DiffusionOpdCampaignReleaseReceipt releaseReceipt,
            DiffusionOpdCampaignReleaseAcknowledgement releaseAcknowledgement,
            DiffusionOpdCampaignReleaseResolution releaseResolution,
            DiffusionOpdCampaignReleaseClosure releaseClosure) {
        String handoffTarget = releaseResolution.escalationTarget();
        boolean terminal = releaseClosure.closed();
        String exportStatus = terminal ? "finalized" : "pending_follow_up";
        String handoffKey = handoffTarget
                + ":"
                + exportStatus
                + ":"
                + releaseClosure.runnableStepCount()
                + ":"
                + releaseClosure.blockedEntryCount();
        return new DiffusionOpdCampaignReleaseHandoffBundle(
                releaseReceipt,
                releaseAcknowledgement,
                releaseResolution,
                releaseClosure,
                handoffTarget,
                exportStatus,
                terminal,
                releaseClosure.runnableStepCount(),
                releaseClosure.blockedEntryCount(),
                handoffKey,
                DiffusionOpdReportQueries.buildCampaignReleaseHandoffBundleMessage(
                        handoffTarget,
                        exportStatus,
                        releaseClosure.runnableStepCount(),
                        releaseClosure.blockedEntryCount()));
    }

    static DiffusionOpdCampaignReleaseExportManifest campaignReleaseExportManifest(
            DiffusionOpdCampaignReleaseHandoffBundle handoffBundle) {
        List<DiffusionOpdCampaignReleaseExportRecord> records = List.of(
                new DiffusionOpdCampaignReleaseExportRecord(
                        "release_receipt",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.releaseReceipt().receiptKey()),
                new DiffusionOpdCampaignReleaseExportRecord(
                        "release_acknowledgement",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.releaseAcknowledgement().acknowledgementKey()),
                new DiffusionOpdCampaignReleaseExportRecord(
                        "release_resolution",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.releaseResolution().resolutionKey()),
                new DiffusionOpdCampaignReleaseExportRecord(
                        "release_closure",
                        handoffBundle.handoffTarget(),
                        handoffBundle.exportStatus(),
                        handoffBundle.releaseClosure().closureKey()));
        return new DiffusionOpdCampaignReleaseExportManifest(
                handoffBundle,
                records,
                records.size(),
                handoffBundle.handoffTarget(),
                handoffBundle.exportStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseExportManifestMessage(
                        records.size(),
                        handoffBundle.exportStatus()));
    }
}
