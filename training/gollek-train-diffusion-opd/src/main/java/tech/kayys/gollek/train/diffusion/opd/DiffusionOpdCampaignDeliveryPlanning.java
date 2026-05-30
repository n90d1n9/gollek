package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignAcknowledgementDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDecisionResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryDestination;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilityEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchEligibilitySummary;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlan;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignDispatchPlanStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReceiptAcknowledgements;

/**
 * Internal delivery-to-dispatch planning logic for campaign workflows.
 *
 * <p>This owns the older campaign delivery, acknowledgement, and dispatch preparation stages after
 * they were split away from the public query surface. Stage-local leaf shaping stays in
 * {@link DiffusionOpdCampaignDeliveryPlanningStages}, while this helper keeps the aggregate stage
 * counts and pipeline-level assembly together.
 */
final class DiffusionOpdCampaignDeliveryPlanning {
    private DiffusionOpdCampaignDeliveryPlanning() {
    }

    static DiffusionOpdCampaignDeliveryReport campaignDeliveryReport(
            DiffusionOpdCampaignExportManifest exportManifest) {
        Map<String, List<DiffusionOpdCampaignExportRecord>> recordsByDestination = exportManifest.records().stream()
                .collect(Collectors.groupingBy(
                        DiffusionOpdCampaignExportRecord::destination,
                        LinkedHashMap::new,
                        Collectors.toList()));
        List<DiffusionOpdCampaignDeliveryDestination> destinations = recordsByDestination.entrySet().stream()
                .map(entry -> DiffusionOpdCampaignDeliveryPlanningStages
                        .campaignDeliveryDestination(entry.getKey(), entry.getValue()))
                .toList();
        int pendingArtifacts = destinations.stream()
                .mapToInt(DiffusionOpdCampaignDeliveryDestination::pendingCount)
                .sum();
        int blockedArtifacts = destinations.stream()
                .mapToInt(DiffusionOpdCampaignDeliveryDestination::blockedCount)
                .sum();
        boolean followUpRequired = destinations.stream()
                .anyMatch(DiffusionOpdCampaignDeliveryDestination::followUpRequired);
        return new DiffusionOpdCampaignDeliveryReport(
                exportManifest,
                destinations,
                destinations.size(),
                exportManifest.recordCount(),
                pendingArtifacts,
                blockedArtifacts,
                followUpRequired,
                exportManifest.primaryDestination(),
                exportManifest.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDeliveryReportMessage(
                        exportManifest.recordCount(),
                        pendingArtifacts,
                        blockedArtifacts));
    }

    static DiffusionOpdCampaignDeliveryLedger campaignDeliveryLedger(
            DiffusionOpdCampaignDeliveryReport deliveryReport) {
        List<DiffusionOpdCampaignDeliveryLedgerEntry> entries = deliveryReport.exportManifest().records().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignDeliveryLedgerEntry)
                .toList();
        int acknowledgedCount = (int) entries.stream()
                .filter(entry -> DiffusionOpdReportQueries.isAcknowledgedDeliveryStatus(entry.acknowledgementStatus()))
                .count();
        int pendingAcknowledgementCount = (int) entries.stream()
                .filter(entry -> DiffusionOpdReportQueries.isPendingAcknowledgementStatus(entry.acknowledgementStatus()))
                .count();
        int retryRequiredCount = (int) entries.stream()
                .filter(entry -> DiffusionOpdReportQueries.requiresRetry(entry.retryPolicy()))
                .count();
        boolean followUpRequired = entries.stream()
                .anyMatch(DiffusionOpdCampaignDeliveryLedgerEntry::followUpRequired);
        return new DiffusionOpdCampaignDeliveryLedger(
                deliveryReport,
                entries,
                entries.size(),
                acknowledgedCount,
                pendingAcknowledgementCount,
                retryRequiredCount,
                followUpRequired,
                deliveryReport.primaryDestination(),
                deliveryReport.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDeliveryLedgerMessage(
                        entries.size(),
                        pendingAcknowledgementCount,
                        retryRequiredCount));
    }

    static DiffusionOpdCampaignDeliveryReceipts campaignDeliveryReceipts(
            DiffusionOpdCampaignDeliveryLedger deliveryLedger) {
        List<DiffusionOpdCampaignDeliveryReceipt> receipts = deliveryLedger.entries().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignDeliveryReceipt)
                .toList();
        int confirmedCount = (int) receipts.stream()
                .filter(receipt -> DiffusionOpdReportQueries.isConfirmedReceiptStatus(receipt.receiptStatus()))
                .count();
        int pendingCount = (int) receipts.stream()
                .filter(receipt -> DiffusionOpdReportQueries.isPendingReceiptStatus(receipt.receiptStatus()))
                .count();
        int operatorActionRequiredCount = (int) receipts.stream()
                .filter(receipt -> "operator_review_required".equals(receipt.operatorAcknowledgement()))
                .count();
        boolean followUpRequired = receipts.stream()
                .anyMatch(DiffusionOpdCampaignDeliveryReceipt::followUpRequired);
        return new DiffusionOpdCampaignDeliveryReceipts(
                deliveryLedger,
                receipts,
                receipts.size(),
                confirmedCount,
                pendingCount,
                operatorActionRequiredCount,
                followUpRequired,
                deliveryLedger.primaryDestination(),
                deliveryLedger.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDeliveryReceiptsMessage(
                        receipts.size(),
                        pendingCount,
                        operatorActionRequiredCount));
    }

    static DiffusionOpdCampaignReceiptAcknowledgements campaignReceiptAcknowledgements(
            DiffusionOpdCampaignDeliveryReceipts deliveryReceipts) {
        List<DiffusionOpdCampaignReceiptAcknowledgement> acknowledgements = deliveryReceipts.receipts().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignReceiptAcknowledgement)
                .toList();
        int acceptedCount = (int) acknowledgements.stream()
                .filter(ack -> "accepted".equals(ack.acknowledgementOutcome()))
                .count();
        int pendingCount = (int) acknowledgements.stream()
                .filter(ack -> "pending_review".equals(ack.acknowledgementOutcome()))
                .count();
        int rejectedCount = (int) acknowledgements.stream()
                .filter(ack -> "rejected".equals(ack.acknowledgementOutcome()))
                .count();
        boolean followUpRequired = acknowledgements.stream()
                .anyMatch(DiffusionOpdCampaignReceiptAcknowledgement::followUpRequired);
        return new DiffusionOpdCampaignReceiptAcknowledgements(
                deliveryReceipts,
                acknowledgements,
                acknowledgements.size(),
                acceptedCount,
                pendingCount,
                rejectedCount,
                followUpRequired,
                deliveryReceipts.primaryDestination(),
                deliveryReceipts.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReceiptAcknowledgementsMessage(
                        acknowledgements.size(),
                        pendingCount,
                        rejectedCount));
    }

    static DiffusionOpdCampaignAcknowledgementDecisions campaignAcknowledgementDecisions(
            DiffusionOpdCampaignReceiptAcknowledgements receiptAcknowledgements) {
        List<DiffusionOpdCampaignAcknowledgementDecision> decisions = receiptAcknowledgements.acknowledgements().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignAcknowledgementDecision)
                .toList();
        int approvedCount = (int) decisions.stream()
                .filter(decision -> "approved".equals(decision.decision()))
                .count();
        int deferredCount = (int) decisions.stream()
                .filter(decision -> "deferred".equals(decision.decision()))
                .count();
        int rejectedCount = (int) decisions.stream()
                .filter(decision -> "rejected".equals(decision.decision()))
                .count();
        boolean followUpRequired = decisions.stream()
                .anyMatch(decision -> !decision.finalDecision());
        return new DiffusionOpdCampaignAcknowledgementDecisions(
                receiptAcknowledgements,
                decisions,
                decisions.size(),
                approvedCount,
                deferredCount,
                rejectedCount,
                followUpRequired,
                receiptAcknowledgements.primaryDestination(),
                receiptAcknowledgements.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignAcknowledgementDecisionsMessage(
                        decisions.size(),
                        deferredCount,
                        rejectedCount));
    }

    static DiffusionOpdCampaignDecisionResolutions campaignDecisionResolutions(
            DiffusionOpdCampaignAcknowledgementDecisions acknowledgementDecisions) {
        List<DiffusionOpdCampaignDecisionResolution> resolutions = acknowledgementDecisions.decisions().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignDecisionResolution)
                .toList();
        int executableCount = (int) resolutions.stream()
                .filter(resolution -> "ready".equals(resolution.executionReadiness()))
                .count();
        int blockedCount = (int) resolutions.stream()
                .filter(resolution -> "blocked".equals(resolution.executionReadiness()))
                .count();
        int escalationRequiredCount = (int) resolutions.stream()
                .filter(resolution -> !"none".equals(resolution.escalationOwner()))
                .count();
        boolean followUpRequired = resolutions.stream()
                .anyMatch(resolution -> !resolution.finalResolution());
        return new DiffusionOpdCampaignDecisionResolutions(
                acknowledgementDecisions,
                resolutions,
                resolutions.size(),
                executableCount,
                blockedCount,
                escalationRequiredCount,
                followUpRequired,
                acknowledgementDecisions.primaryDestination(),
                acknowledgementDecisions.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDecisionResolutionsMessage(
                        resolutions.size(),
                        blockedCount,
                        escalationRequiredCount));
    }

    static DiffusionOpdCampaignDispatchEligibilitySummary campaignDispatchEligibilitySummary(
            DiffusionOpdCampaignDecisionResolutions decisionResolutions) {
        List<DiffusionOpdCampaignDispatchEligibilityEntry> entries = decisionResolutions.resolutions().stream()
                .map(DiffusionOpdCampaignDeliveryPlanningStages::campaignDispatchEligibilityEntry)
                .toList();
        int dispatchableCount = (int) entries.stream()
                .filter(DiffusionOpdCampaignDispatchEligibilityEntry::dispatchable)
                .count();
        int blockedCount = (int) entries.stream()
                .filter(entry -> !entry.dispatchable())
                .count();
        int escalationRequiredCount = (int) entries.stream()
                .filter(entry -> !"none".equals(entry.escalationOwner()))
                .count();
        return new DiffusionOpdCampaignDispatchEligibilitySummary(
                decisionResolutions,
                entries,
                entries.size(),
                dispatchableCount,
                blockedCount,
                escalationRequiredCount,
                dispatchableCount > 0,
                decisionResolutions.primaryDestination(),
                decisionResolutions.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDispatchEligibilitySummaryMessage(
                        entries.size(),
                        dispatchableCount,
                        escalationRequiredCount));
    }

    static DiffusionOpdCampaignDispatchPlan campaignDispatchPlan(
            DiffusionOpdCampaignDispatchEligibilitySummary eligibilitySummary) {
        List<DiffusionOpdCampaignDispatchPlanStep> steps = eligibilitySummary.entries().stream()
                .filter(DiffusionOpdCampaignDispatchEligibilityEntry::dispatchable)
                .map(entry -> DiffusionOpdCampaignDeliveryPlanningStages
                        .campaignDispatchPlanStep(entry, eligibilitySummary.primaryDestination()))
                .toList();
        return new DiffusionOpdCampaignDispatchPlan(
                eligibilitySummary,
                steps,
                steps.size(),
                eligibilitySummary.blockedCount(),
                eligibilitySummary.escalationRequiredCount(),
                !steps.isEmpty(),
                eligibilitySummary.primaryDestination(),
                eligibilitySummary.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignDispatchPlanMessage(
                        steps.size(),
                        eligibilitySummary.blockedCount(),
                        eligibilitySummary.escalationRequiredCount()));
    }

}
