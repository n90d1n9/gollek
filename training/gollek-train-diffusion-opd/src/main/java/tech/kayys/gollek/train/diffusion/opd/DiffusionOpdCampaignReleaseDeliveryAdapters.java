package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;

/**
 * Compatibility projection layer for release-delivery types.
 *
 * <p>This adapts the canonical export-lane objects into the older aggregate release-delivery public
 * models without reintroducing a second implementation of the delivery pipeline. It should stay a
 * pure projection layer: canonical export objects come in, compatibility-shaped delivery objects
 * come out, and no new delivery-state decisions are introduced here.
 */
final class DiffusionOpdCampaignReleaseDeliveryAdapters {
    private DiffusionOpdCampaignReleaseDeliveryAdapters() {
    }

    static CampaignReleaseDeliveryAdapterPipeline buildCampaignReleaseDeliveryAdapterPipeline(
            DiffusionOpdCampaignReleaseExportManifest exportManifest,
            DiffusionOpdCampaignReleaseDeliveryExports.CampaignReleaseDeliveryExportPipeline exportPipeline) {
        DiffusionOpdCampaignReleaseDeliveryReport report = toCampaignReleaseDeliveryReport(
                exportManifest,
                exportPipeline.exportReport());
        DiffusionOpdCampaignReleaseDeliveryLedger ledger = toCampaignReleaseDeliveryLedger(
                report,
                exportPipeline.exportLedger());
        DiffusionOpdCampaignReleaseDeliveryReceipts receipts = toCampaignReleaseDeliveryReceipts(
                ledger,
                exportPipeline.exportReceipts());
        DiffusionOpdCampaignReleaseDeliveryAcknowledgements acknowledgements = toCampaignReleaseDeliveryAcknowledgements(
                receipts,
                exportPipeline.exportAcknowledgements());
        DiffusionOpdCampaignReleaseDeliveryDecisions decisions = toCampaignReleaseDeliveryDecisions(
                acknowledgements,
                exportPipeline.exportDecisions());
        DiffusionOpdCampaignReleaseDeliveryResolutions resolutions = toCampaignReleaseDeliveryResolutions(
                decisions,
                exportPipeline.exportResolutions());
        DiffusionOpdCampaignReleaseDeliveryClosures closures = toCampaignReleaseDeliveryClosures(
                resolutions,
                exportPipeline.exportClosures());
        List<DiffusionOpdCampaignReleaseDeliveryHandoffBundle> handoffBundles = exportPipeline.handoffBundles().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryHandoffBundle)
                .toList();
        return new CampaignReleaseDeliveryAdapterPipeline(
                report,
                ledger,
                receipts,
                acknowledgements,
                decisions,
                resolutions,
                closures,
                handoffBundles);
    }

    static DiffusionOpdCampaignReleaseDeliveryReport toCampaignReleaseDeliveryReport(
            DiffusionOpdCampaignReleaseExportManifest exportManifest,
            DiffusionOpdCampaignReleaseDeliveryExportReport exportReport) {
        return new DiffusionOpdCampaignReleaseDeliveryReport(
                exportManifest,
                exportReport.destinations(),
                exportReport.destinationCount(),
                exportReport.totalArtifacts(),
                exportReport.pendingArtifacts(),
                exportReport.blockedArtifacts(),
                exportReport.followUpRequired(),
                exportReport.primaryDestination(),
                exportReport.overallDeliveryStatus(),
                exportReport.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryLedger toCampaignReleaseDeliveryLedger(
            DiffusionOpdCampaignReleaseDeliveryReport deliveryReport,
            DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger) {
        List<DiffusionOpdCampaignReleaseDeliveryLedgerEntry> entries = exportLedger.entries().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryLedgerEntry)
                .toList();
        return new DiffusionOpdCampaignReleaseDeliveryLedger(
                deliveryReport,
                entries,
                exportLedger.entryCount(),
                exportLedger.acknowledgedCount(),
                exportLedger.pendingAcknowledgementCount(),
                exportLedger.retryRequiredCount(),
                exportLedger.followUpRequired(),
                exportLedger.primaryDestination(),
                exportLedger.overallDeliveryStatus(),
                exportLedger.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryReceipts toCampaignReleaseDeliveryReceipts(
            DiffusionOpdCampaignReleaseDeliveryLedger deliveryLedger,
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts) {
        List<DiffusionOpdCampaignReleaseDeliveryReceipt> receipts = exportReceipts.receipts().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryReceipt)
                .toList();
        return new DiffusionOpdCampaignReleaseDeliveryReceipts(
                deliveryLedger,
                receipts,
                exportReceipts.receiptCount(),
                exportReceipts.confirmedCount(),
                exportReceipts.pendingCount(),
                exportReceipts.operatorActionRequiredCount(),
                exportReceipts.followUpRequired(),
                exportReceipts.primaryDestination(),
                exportReceipts.overallDeliveryStatus(),
                exportReceipts.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryAcknowledgements toCampaignReleaseDeliveryAcknowledgements(
            DiffusionOpdCampaignReleaseDeliveryReceipts deliveryReceipts,
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements) {
        List<DiffusionOpdCampaignReleaseDeliveryAcknowledgement> acknowledgements =
                exportAcknowledgements.acknowledgements().stream()
                        .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryAcknowledgement)
                        .toList();
        return new DiffusionOpdCampaignReleaseDeliveryAcknowledgements(
                deliveryReceipts,
                acknowledgements,
                exportAcknowledgements.acknowledgementCount(),
                exportAcknowledgements.acceptedCount(),
                exportAcknowledgements.pendingCount(),
                exportAcknowledgements.rejectedCount(),
                exportAcknowledgements.followUpRequired(),
                exportAcknowledgements.primaryDestination(),
                exportAcknowledgements.overallDeliveryStatus(),
                exportAcknowledgements.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryDecisions toCampaignReleaseDeliveryDecisions(
            DiffusionOpdCampaignReleaseDeliveryAcknowledgements deliveryAcknowledgements,
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions) {
        List<DiffusionOpdCampaignReleaseDeliveryDecision> decisions = exportDecisions.decisions().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryDecision)
                .toList();
        return new DiffusionOpdCampaignReleaseDeliveryDecisions(
                deliveryAcknowledgements,
                decisions,
                exportDecisions.decisionCount(),
                exportDecisions.acceptedCount(),
                exportDecisions.deferredCount(),
                exportDecisions.rejectedCount(),
                exportDecisions.followUpRequired(),
                exportDecisions.primaryDestination(),
                exportDecisions.overallDeliveryStatus(),
                exportDecisions.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryResolutions toCampaignReleaseDeliveryResolutions(
            DiffusionOpdCampaignReleaseDeliveryDecisions deliveryDecisions,
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions) {
        List<DiffusionOpdCampaignReleaseDeliveryResolution> resolutions = exportResolutions.resolutions().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryResolution)
                .toList();
        return new DiffusionOpdCampaignReleaseDeliveryResolutions(
                deliveryDecisions,
                resolutions,
                exportResolutions.resolutionCount(),
                exportResolutions.acceptedCount(),
                exportResolutions.deferredCount(),
                exportResolutions.rejectedCount(),
                exportResolutions.followUpRequired(),
                exportResolutions.primaryDestination(),
                exportResolutions.overallDeliveryStatus(),
                exportResolutions.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryClosures toCampaignReleaseDeliveryClosures(
            DiffusionOpdCampaignReleaseDeliveryResolutions deliveryResolutions,
            DiffusionOpdCampaignReleaseDeliveryExportClosures exportClosures) {
        List<DiffusionOpdCampaignReleaseDeliveryClosure> closures = exportClosures.closures().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryAdapters::toCampaignReleaseDeliveryClosure)
                .toList();
        return new DiffusionOpdCampaignReleaseDeliveryClosures(
                deliveryResolutions,
                closures,
                exportClosures.closureCount(),
                exportClosures.closedCount(),
                exportClosures.openCount(),
                exportClosures.followUpRequired(),
                exportClosures.primaryDestination(),
                exportClosures.overallDeliveryStatus(),
                exportClosures.summaryMessage());
    }

    static DiffusionOpdCampaignReleaseDeliveryHandoffBundle toCampaignReleaseDeliveryHandoffBundle(
            DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle exportBundle) {
        return new DiffusionOpdCampaignReleaseDeliveryHandoffBundle(
                toCampaignReleaseDeliveryReceipt(exportBundle.receipt()),
                toCampaignReleaseDeliveryAcknowledgement(exportBundle.acknowledgement()),
                toCampaignReleaseDeliveryDecision(exportBundle.decision()),
                toCampaignReleaseDeliveryResolution(exportBundle.resolution()),
                toCampaignReleaseDeliveryClosure(exportBundle.closure()),
                exportBundle.handoffTarget(),
                exportBundle.exportStatus(),
                exportBundle.terminal(),
                exportBundle.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryLedgerEntry toCampaignReleaseDeliveryLedgerEntry(
            DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry exportEntry) {
        return new DiffusionOpdCampaignReleaseDeliveryLedgerEntry(
                exportEntry.artifactType(),
                exportEntry.destination(),
                exportEntry.referenceKey(),
                exportEntry.deliveryStatus(),
                exportEntry.acknowledgementStatus(),
                exportEntry.retryPolicy(),
                exportEntry.attemptCount(),
                exportEntry.followUpRequired(),
                exportEntry.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryReceipt toCampaignReleaseDeliveryReceipt(
            DiffusionOpdCampaignReleaseDeliveryExportReceipt exportReceipt) {
        return new DiffusionOpdCampaignReleaseDeliveryReceipt(
                exportReceipt.artifactType(),
                exportReceipt.destination(),
                exportReceipt.referenceKey(),
                exportReceipt.receiptId(),
                exportReceipt.receiptStatus(),
                exportReceipt.acknowledgementStatus(),
                exportReceipt.operatorAcknowledgement(),
                exportReceipt.deliveryTimestamp(),
                exportReceipt.followUpRequired(),
                exportReceipt.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryAcknowledgement toCampaignReleaseDeliveryAcknowledgement(
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement exportAcknowledgement) {
        return new DiffusionOpdCampaignReleaseDeliveryAcknowledgement(
                exportAcknowledgement.receiptId(),
                exportAcknowledgement.artifactType(),
                exportAcknowledgement.destination(),
                exportAcknowledgement.acknowledgementOutcome(),
                exportAcknowledgement.reviewer(),
                exportAcknowledgement.reviewerRole(),
                exportAcknowledgement.finalAcknowledgement(),
                exportAcknowledgement.followUpRequired(),
                exportAcknowledgement.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryDecision toCampaignReleaseDeliveryDecision(
            DiffusionOpdCampaignReleaseDeliveryExportDecision exportDecision) {
        return new DiffusionOpdCampaignReleaseDeliveryDecision(
                exportDecision.receiptId(),
                exportDecision.artifactType(),
                exportDecision.destination(),
                exportDecision.decision(),
                exportDecision.reviewer(),
                exportDecision.reviewerRole(),
                exportDecision.resolutionRoute(),
                exportDecision.finalDecision(),
                exportDecision.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryResolution toCampaignReleaseDeliveryResolution(
            DiffusionOpdCampaignReleaseDeliveryExportResolution exportResolution) {
        return new DiffusionOpdCampaignReleaseDeliveryResolution(
                exportResolution.receiptId(),
                exportResolution.artifactType(),
                exportResolution.destination(),
                exportResolution.resolution(),
                exportResolution.escalationTarget(),
                exportResolution.finalAccepted(),
                exportResolution.downstreamEligible(),
                exportResolution.summaryMessage());
    }

    private static DiffusionOpdCampaignReleaseDeliveryClosure toCampaignReleaseDeliveryClosure(
            DiffusionOpdCampaignReleaseDeliveryExportClosure exportClosure) {
        return new DiffusionOpdCampaignReleaseDeliveryClosure(
                exportClosure.receiptId(),
                exportClosure.artifactType(),
                exportClosure.destination(),
                exportClosure.finalOutcome(),
                exportClosure.closed(),
                exportClosure.followUpAction(),
                exportClosure.dispatchComplete(),
                exportClosure.summaryMessage());
    }

    record CampaignReleaseDeliveryAdapterPipeline(
            DiffusionOpdCampaignReleaseDeliveryReport report,
            DiffusionOpdCampaignReleaseDeliveryLedger ledger,
            DiffusionOpdCampaignReleaseDeliveryReceipts receipts,
            DiffusionOpdCampaignReleaseDeliveryAcknowledgements acknowledgements,
            DiffusionOpdCampaignReleaseDeliveryDecisions decisions,
            DiffusionOpdCampaignReleaseDeliveryResolutions resolutions,
            DiffusionOpdCampaignReleaseDeliveryClosures closures,
            List<DiffusionOpdCampaignReleaseDeliveryHandoffBundle> handoffBundles) {
    }
}
