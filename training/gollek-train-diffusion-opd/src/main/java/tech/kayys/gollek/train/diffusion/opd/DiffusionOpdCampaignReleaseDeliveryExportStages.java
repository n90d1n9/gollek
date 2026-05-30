package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryDestination;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolution;

/**
 * Owns the per-stage export leaf projections so the main export helper can stay
 * focused on pipeline assembly.
 *
 * <p>This file should stay limited to stage-local object shaping. It should not decide pipeline
 * order, rebuild export roots, or reintroduce cross-stage orchestration.
 */
final class DiffusionOpdCampaignReleaseDeliveryExportStages {

    private DiffusionOpdCampaignReleaseDeliveryExportStages() {
    }

    static DiffusionOpdCampaignReleaseDeliveryDestination campaignReleaseDeliveryExportDestination(
            String destination,
            List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records) {
        List<String> artifactTypes = records.stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportRecord::artifactType)
                .toList();
        int blockedCount = records.isEmpty() ? 0 : records.size();
        return new DiffusionOpdCampaignReleaseDeliveryDestination(
                destination,
                artifactTypes,
                records.size(),
                records.size(),
                blockedCount,
                blockedCount > 0,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryDestinationMessage(
                        destination,
                        records.size(),
                        blockedCount));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry campaignReleaseDeliveryExportLedgerEntry(
            DiffusionOpdCampaignReleaseDeliveryExportRecord record) {
        return new DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry(
                record.artifactType(),
                record.destination(),
                record.referenceKey(),
                record.deliveryStatus(),
                "pending_ack",
                "await_follow_up",
                0,
                true,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportLedgerEntryMessage(
                        record.artifactType(),
                        record.destination(),
                        record.deliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportReceipt campaignReleaseDeliveryExportReceipt(
            DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry entry) {
        return new DiffusionOpdCampaignReleaseDeliveryExportReceipt(
                entry.artifactType(),
                entry.destination(),
                entry.referenceKey(),
                entry.referenceKey() + ":receipt",
                "pending_confirmation",
                entry.acknowledgementStatus(),
                "awaiting_operator_confirmation",
                "pending",
                entry.followUpRequired(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportReceiptMessage(
                        entry.artifactType(),
                        entry.destination(),
                        entry.deliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement campaignReleaseDeliveryExportAcknowledgement(
            DiffusionOpdCampaignReleaseDeliveryExportReceipt receipt) {
        return new DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement(
                receipt.receiptId(),
                receipt.artifactType(),
                receipt.destination(),
                "pending_review",
                "release-bot",
                "operator-gate",
                false,
                receipt.followUpRequired(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportAcknowledgementMessage(
                        receipt.artifactType(),
                        receipt.destination(),
                        receipt.receiptStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportDecision campaignReleaseDeliveryExportDecision(
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement acknowledgement) {
        return new DiffusionOpdCampaignReleaseDeliveryExportDecision(
                acknowledgement.receiptId(),
                acknowledgement.artifactType(),
                acknowledgement.destination(),
                "deferred",
                acknowledgement.reviewer(),
                acknowledgement.reviewerRole(),
                "review_queue",
                false,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportDecisionMessage(
                        acknowledgement.artifactType(),
                        acknowledgement.destination(),
                        "deferred"));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportResolution campaignReleaseDeliveryExportResolution(
            DiffusionOpdCampaignReleaseDeliveryExportDecision decision) {
        return new DiffusionOpdCampaignReleaseDeliveryExportResolution(
                decision.receiptId(),
                decision.artifactType(),
                decision.destination(),
                "deferred",
                "training-monitor",
                false,
                false,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportResolutionMessage(
                        decision.artifactType(),
                        decision.destination(),
                        "deferred"));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportClosure campaignReleaseDeliveryExportClosure(
            DiffusionOpdCampaignReleaseDeliveryExportResolution resolution) {
        return new DiffusionOpdCampaignReleaseDeliveryExportClosure(
                resolution.receiptId(),
                resolution.artifactType(),
                resolution.destination(),
                "pending_follow_up",
                false,
                "await_manual_review",
                false,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportClosureMessage(
                        resolution.artifactType(),
                        resolution.destination(),
                        "pending_follow_up"));
    }
}
