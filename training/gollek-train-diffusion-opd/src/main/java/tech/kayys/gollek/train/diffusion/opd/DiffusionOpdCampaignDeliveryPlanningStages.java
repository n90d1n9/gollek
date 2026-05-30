package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryDestination;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilityEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlanStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgement;

/**
 * Owns the per-stage delivery-to-dispatch projections so
 * {@link DiffusionOpdCampaignDeliveryPlanning} can stay focused on pipeline aggregation, counts,
 * and status rollups.
 */
final class DiffusionOpdCampaignDeliveryPlanningStages {

    private DiffusionOpdCampaignDeliveryPlanningStages() {
    }

    static DiffusionOpdCampaignDeliveryDestination campaignDeliveryDestination(
            String destination,
            List<DiffusionOpdCampaignExportRecord> records) {
        int deliveredCount = (int) records.stream()
                .filter(record -> DiffusionOpdReportQueries.isDeliveredDeliveryStatus(record.deliveryStatus()))
                .count();
        int pendingCount = (int) records.stream()
                .filter(record -> DiffusionOpdReportQueries.isPendingDeliveryStatus(record.deliveryStatus()))
                .count();
        int blockedCount = (int) records.stream()
                .filter(record -> DiffusionOpdReportQueries.isBlockedDeliveryStatus(record.deliveryStatus()))
                .count();
        boolean followUpRequired = pendingCount > 0 || blockedCount > 0;
        List<String> artifactTypes = records.stream()
                .map(DiffusionOpdCampaignExportRecord::artifactType)
                .distinct()
                .toList();
        return new DiffusionOpdCampaignDeliveryDestination(
                destination,
                artifactTypes,
                records.size(),
                deliveredCount,
                pendingCount,
                blockedCount,
                followUpRequired,
                DiffusionOpdReportQueries.buildCampaignDeliveryDestinationMessage(
                        destination,
                        records.size(),
                        pendingCount,
                        blockedCount));
    }

    static DiffusionOpdCampaignDeliveryLedgerEntry campaignDeliveryLedgerEntry(
            DiffusionOpdCampaignExportRecord record) {
        String acknowledgementStatus = DiffusionOpdReportQueries.acknowledgementStatus(record.deliveryStatus());
        String retryPolicy = DiffusionOpdReportQueries.deliveryRetryPolicy(record.deliveryStatus());
        boolean followUpRequired = DiffusionOpdReportQueries.isPendingAcknowledgementStatus(acknowledgementStatus)
                || DiffusionOpdReportQueries.requiresRetry(retryPolicy);
        return new DiffusionOpdCampaignDeliveryLedgerEntry(
                record.artifactType(),
                record.destination(),
                record.referenceKey(),
                record.deliveryStatus(),
                acknowledgementStatus,
                retryPolicy,
                1,
                followUpRequired,
                DiffusionOpdReportQueries.buildCampaignDeliveryLedgerEntryMessage(
                        record.artifactType(),
                        record.destination(),
                        acknowledgementStatus,
                        retryPolicy));
    }

    static DiffusionOpdCampaignDeliveryReceipt campaignDeliveryReceipt(
            DiffusionOpdCampaignDeliveryLedgerEntry entry) {
        String receiptStatus = DiffusionOpdReportQueries.receiptStatus(entry.acknowledgementStatus());
        String operatorAcknowledgement = DiffusionOpdReportQueries.operatorAcknowledgement(
                entry.acknowledgementStatus(),
                entry.retryPolicy());
        return new DiffusionOpdCampaignDeliveryReceipt(
                entry.artifactType(),
                entry.destination(),
                entry.referenceKey(),
                DiffusionOpdReportQueries.deliveryReceiptId(entry.referenceKey(), entry.artifactType()),
                receiptStatus,
                entry.acknowledgementStatus(),
                operatorAcknowledgement,
                DiffusionOpdReportQueries.deliveryTimestamp(entry.deliveryStatus()),
                entry.followUpRequired(),
                DiffusionOpdReportQueries.buildCampaignDeliveryReceiptMessage(
                        entry.artifactType(),
                        entry.destination(),
                        receiptStatus,
                        operatorAcknowledgement));
    }

    static DiffusionOpdCampaignReceiptAcknowledgement campaignReceiptAcknowledgement(
            DiffusionOpdCampaignDeliveryReceipt receipt) {
        String outcome = DiffusionOpdReportQueries.receiptAcknowledgementOutcome(
                receipt.receiptStatus(),
                receipt.operatorAcknowledgement());
        boolean finalAcknowledgement = DiffusionOpdReportQueries.isFinalAcknowledgementOutcome(outcome);
        boolean followUpRequired = !finalAcknowledgement;
        return new DiffusionOpdCampaignReceiptAcknowledgement(
                receipt.receiptId(),
                receipt.artifactType(),
                receipt.destination(),
                outcome,
                DiffusionOpdReportQueries.receiptReviewer(receipt.operatorAcknowledgement()),
                DiffusionOpdReportQueries.receiptReviewerRole(receipt.operatorAcknowledgement()),
                finalAcknowledgement,
                followUpRequired,
                DiffusionOpdReportQueries.buildCampaignReceiptAcknowledgementMessage(
                        receipt.receiptId(),
                        outcome,
                        DiffusionOpdReportQueries.receiptReviewer(receipt.operatorAcknowledgement())));
    }

    static DiffusionOpdCampaignAcknowledgementDecision campaignAcknowledgementDecision(
            DiffusionOpdCampaignReceiptAcknowledgement acknowledgement) {
        String decision = DiffusionOpdReportQueries.acknowledgementDecision(acknowledgement.acknowledgementOutcome());
        return new DiffusionOpdCampaignAcknowledgementDecision(
                acknowledgement.receiptId(),
                acknowledgement.artifactType(),
                decision,
                acknowledgement.reviewer(),
                acknowledgement.reviewerRole(),
                DiffusionOpdReportQueries.acknowledgementDecisionNote(decision),
                DiffusionOpdReportQueries.acknowledgementResolutionRoute(decision),
                DiffusionOpdReportQueries.isFinalAcknowledgementDecision(decision),
                DiffusionOpdReportQueries.buildCampaignAcknowledgementDecisionMessage(
                        acknowledgement.receiptId(),
                        decision,
                        DiffusionOpdReportQueries.acknowledgementResolutionRoute(decision)));
    }

    static DiffusionOpdCampaignDecisionResolution campaignDecisionResolution(
            DiffusionOpdCampaignAcknowledgementDecision decision) {
        String resolution = DiffusionOpdReportQueries.campaignDecisionResolutionValue(decision.decision());
        return new DiffusionOpdCampaignDecisionResolution(
                decision.receiptId(),
                decision.artifactType(),
                resolution,
                DiffusionOpdReportQueries.campaignEscalationOwner(resolution),
                DiffusionOpdReportQueries.campaignExecutionReadiness(resolution),
                DiffusionOpdReportQueries.campaignDownstreamEligibility(resolution),
                DiffusionOpdReportQueries.isFinalCampaignDecisionResolution(resolution),
                DiffusionOpdReportQueries.buildCampaignDecisionResolutionMessage(
                        decision.receiptId(),
                        resolution,
                        DiffusionOpdReportQueries.campaignExecutionReadiness(resolution)));
    }

    static DiffusionOpdCampaignDispatchEligibilityEntry campaignDispatchEligibilityEntry(
            DiffusionOpdCampaignDecisionResolution resolution) {
        String eligibility = "eligible".equals(resolution.downstreamEligibility())
                ? "dispatchable"
                : "blocked";
        boolean dispatchable = "dispatchable".equals(eligibility) && "ready".equals(resolution.executionReadiness());
        return new DiffusionOpdCampaignDispatchEligibilityEntry(
                resolution.receiptId(),
                resolution.artifactType(),
                eligibility,
                resolution.executionReadiness(),
                resolution.escalationOwner(),
                dispatchable,
                DiffusionOpdReportQueries.buildCampaignDispatchEligibilityEntryMessage(
                        resolution.receiptId(),
                        eligibility,
                        resolution.executionReadiness()));
    }

    static DiffusionOpdCampaignDispatchPlanStep campaignDispatchPlanStep(
            DiffusionOpdCampaignDispatchEligibilityEntry entry,
            String primaryDestination) {
        return new DiffusionOpdCampaignDispatchPlanStep(
                1,
                entry.receiptId(),
                entry.artifactType(),
                "automatic",
                primaryDestination,
                DiffusionOpdReportQueries.buildCampaignDispatchPlanStepMessage(
                        entry.receiptId(),
                        "automatic",
                        primaryDestination));
    }
}
