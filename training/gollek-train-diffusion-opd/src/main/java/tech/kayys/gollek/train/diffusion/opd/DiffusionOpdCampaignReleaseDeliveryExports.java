package tech.kayys.gollek.train.diffusion.opd;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryDestination;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosure;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportClosures;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecision;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportDecisions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedger;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipt;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReceipts;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportRecord;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolution;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseDeliveryExportResolutions;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportRecord;

/**
 * Canonical release-delivery export lane for campaign workflows.
 *
 * <p>This helper owns the dedicated export pipeline and its staged export artifacts; compatibility
 * projections back to the older public release-delivery types live elsewhere. The export manifest
 * is the single root input for this lane, and this helper is responsible for carrying it forward
 * into the export report, ledger, receipts, decisions, resolutions, closures, and handoff bundles.
 */
final class DiffusionOpdCampaignReleaseDeliveryExports {
    private DiffusionOpdCampaignReleaseDeliveryExports() {
    }

    static CampaignReleaseDeliveryExportPipeline buildCampaignReleaseDeliveryExportPipeline(
            DiffusionOpdCampaignReleaseExportManifest exportManifest) {
        List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records = exportManifest.records().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExports::toCampaignReleaseDeliveryExportRecord)
                .toList();
        DiffusionOpdCampaignReleaseDeliveryExportManifest seededManifest =
                campaignReleaseDeliveryExportSeedManifest(
                        records,
                        exportManifest.primaryDestination(),
                        exportManifest.overallDeliveryStatus());
        DiffusionOpdCampaignReleaseDeliveryExportReport exportReport =
                campaignReleaseDeliveryExportReport(seededManifest);
        DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger =
                campaignReleaseDeliveryExportLedger(exportReport);
        DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts =
                campaignReleaseDeliveryExportReceipts(exportLedger);
        DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements =
                campaignReleaseDeliveryExportAcknowledgements(exportReceipts);
        DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions =
                campaignReleaseDeliveryExportDecisions(exportAcknowledgements);
        DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions =
                campaignReleaseDeliveryExportResolutions(exportDecisions);
        DiffusionOpdCampaignReleaseDeliveryExportClosures exportClosures =
                campaignReleaseDeliveryExportClosures(exportResolutions);
        List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> seededHandoffBundles =
                buildCampaignReleaseDeliveryExportHandoffBundles(
                        exportReceipts,
                        exportAcknowledgements,
                        exportDecisions,
                        exportResolutions,
                        exportClosures);
        DiffusionOpdCampaignReleaseDeliveryExportManifest exportDeliveryManifest =
                campaignReleaseDeliveryExportManifest(seededHandoffBundles);
        DiffusionOpdCampaignReleaseDeliveryExportReport finalExportReport =
                campaignReleaseDeliveryExportReport(exportDeliveryManifest);
        DiffusionOpdCampaignReleaseDeliveryExportLedger finalExportLedger =
                campaignReleaseDeliveryExportLedger(finalExportReport);
        DiffusionOpdCampaignReleaseDeliveryExportReceipts finalExportReceipts =
                campaignReleaseDeliveryExportReceipts(finalExportLedger);
        DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements finalExportAcknowledgements =
                campaignReleaseDeliveryExportAcknowledgements(finalExportReceipts);
        DiffusionOpdCampaignReleaseDeliveryExportDecisions finalExportDecisions =
                campaignReleaseDeliveryExportDecisions(finalExportAcknowledgements);
        DiffusionOpdCampaignReleaseDeliveryExportResolutions finalExportResolutions =
                campaignReleaseDeliveryExportResolutions(finalExportDecisions);
        DiffusionOpdCampaignReleaseDeliveryExportClosures finalExportClosures =
                campaignReleaseDeliveryExportClosures(finalExportResolutions);
        List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> finalHandoffBundles =
                buildCampaignReleaseDeliveryExportHandoffBundles(
                        finalExportReceipts,
                        finalExportAcknowledgements,
                        finalExportDecisions,
                        finalExportResolutions,
                        finalExportClosures);
        return new CampaignReleaseDeliveryExportPipeline(
                exportDeliveryManifest,
                finalExportReport,
                finalExportLedger,
                finalExportReceipts,
                finalExportAcknowledgements,
                finalExportDecisions,
                finalExportResolutions,
                finalExportClosures,
                finalHandoffBundles);
    }

    static DiffusionOpdCampaignReleaseDeliveryExportRecord toCampaignReleaseDeliveryExportRecord(
            DiffusionOpdCampaignReleaseExportRecord record) {
        return new DiffusionOpdCampaignReleaseDeliveryExportRecord(
                record.artifactType(),
                record.destination(),
                record.deliveryStatus(),
                record.referenceKey());
    }

    static DiffusionOpdCampaignReleaseDeliveryExportManifest campaignReleaseDeliveryExportSeedManifest(
            List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records,
            String primaryDestination,
            String overallDeliveryStatus) {
        return new DiffusionOpdCampaignReleaseDeliveryExportManifest(
                List.of(),
                records,
                records.size(),
                primaryDestination,
                overallDeliveryStatus,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportManifestMessage(
                        records.size(),
                        overallDeliveryStatus));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportManifest campaignReleaseDeliveryExportManifest(
            List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> handoffBundles) {
        List<DiffusionOpdCampaignReleaseDeliveryExportRecord> records = handoffBundles.stream()
                .map(bundle -> new DiffusionOpdCampaignReleaseDeliveryExportRecord(
                        bundle.receipt().artifactType(),
                        bundle.handoffTarget(),
                        bundle.exportStatus(),
                        bundle.receipt().receiptId()))
                .toList();
        String primaryDestination = handoffBundles.isEmpty() ? "none" : handoffBundles.getFirst().handoffTarget();
        String overallDeliveryStatus = handoffBundles.isEmpty() ? "empty" : handoffBundles.getFirst().exportStatus();
        return new DiffusionOpdCampaignReleaseDeliveryExportManifest(
                handoffBundles,
                records,
                records.size(),
                primaryDestination,
                overallDeliveryStatus,
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportManifestMessage(
                        records.size(),
                        overallDeliveryStatus));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportReport campaignReleaseDeliveryExportReport(
            DiffusionOpdCampaignReleaseDeliveryExportManifest exportManifest) {
        Map<String, List<DiffusionOpdCampaignReleaseDeliveryExportRecord>> recordsByDestination =
                exportManifest.records().stream()
                        .collect(Collectors.groupingBy(
                                DiffusionOpdCampaignReleaseDeliveryExportRecord::destination,
                                LinkedHashMap::new,
                                Collectors.toList()));
        List<DiffusionOpdCampaignReleaseDeliveryDestination> destinations = recordsByDestination.entrySet().stream()
                .map(entry -> DiffusionOpdCampaignReleaseDeliveryExportStages
                        .campaignReleaseDeliveryExportDestination(entry.getKey(), entry.getValue()))
                .toList();
        int pendingArtifacts = exportManifest.recordCount();
        int blockedArtifacts =
                "finalized".equals(exportManifest.overallDeliveryStatus()) ? 0 : exportManifest.recordCount();
        boolean followUpRequired = destinations.stream()
                .anyMatch(DiffusionOpdCampaignReleaseDeliveryDestination::followUpRequired);
        return new DiffusionOpdCampaignReleaseDeliveryExportReport(
                exportManifest,
                destinations,
                destinations.size(),
                exportManifest.recordCount(),
                pendingArtifacts,
                blockedArtifacts,
                followUpRequired,
                exportManifest.primaryDestination(),
                exportManifest.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportReportMessage(
                        exportManifest.recordCount(),
                        pendingArtifacts,
                        blockedArtifacts,
                        exportManifest.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportLedger campaignReleaseDeliveryExportLedger(
            DiffusionOpdCampaignReleaseDeliveryExportReport exportReport) {
        List<DiffusionOpdCampaignReleaseDeliveryExportLedgerEntry> entries = exportReport.exportManifest().records().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportLedgerEntry)
                .toList();
        int pendingAcknowledgementCount = entries.stream()
                .mapToInt(entry -> "pending_ack".equals(entry.acknowledgementStatus()) ? 1 : 0)
                .sum();
        int retryRequiredCount = entries.stream()
                .mapToInt(entry -> entry.followUpRequired() ? 1 : 0)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportLedger(
                exportReport,
                entries,
                entries.size(),
                0,
                pendingAcknowledgementCount,
                retryRequiredCount,
                exportReport.followUpRequired(),
                exportReport.primaryDestination(),
                exportReport.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportLedgerMessage(
                        entries.size(),
                        pendingAcknowledgementCount,
                        retryRequiredCount,
                        exportReport.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportReceipts campaignReleaseDeliveryExportReceipts(
            DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger) {
        List<DiffusionOpdCampaignReleaseDeliveryExportReceipt> receipts = exportLedger.entries().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportReceipt)
                .toList();
        int pendingCount = receipts.stream()
                .mapToInt(receipt -> "pending_confirmation".equals(receipt.receiptStatus()) ? 1 : 0)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportReceipts(
                exportLedger,
                receipts,
                receipts.size(),
                0,
                pendingCount,
                pendingCount,
                exportLedger.followUpRequired(),
                exportLedger.primaryDestination(),
                exportLedger.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportReceiptsMessage(
                        receipts.size(),
                        pendingCount,
                        exportLedger.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements campaignReleaseDeliveryExportAcknowledgements(
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts) {
        List<DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement> acknowledgements =
                exportReceipts.receipts().stream()
                        .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportAcknowledgement)
                        .toList();
        int pendingCount = acknowledgements.stream()
                .mapToInt(ack -> "pending_review".equals(ack.acknowledgementOutcome()) ? 1 : 0)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements(
                exportReceipts,
                acknowledgements,
                acknowledgements.size(),
                0,
                pendingCount,
                0,
                exportReceipts.followUpRequired(),
                exportReceipts.primaryDestination(),
                exportReceipts.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportAcknowledgementsMessage(
                        acknowledgements.size(),
                        pendingCount,
                        exportReceipts.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportDecisions campaignReleaseDeliveryExportDecisions(
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements) {
        List<DiffusionOpdCampaignReleaseDeliveryExportDecision> decisions = exportAcknowledgements.acknowledgements().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportDecision)
                .toList();
        int deferredCount = decisions.stream()
                .mapToInt(decision -> "deferred".equals(decision.decision()) ? 1 : 0)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportDecisions(
                exportAcknowledgements,
                decisions,
                decisions.size(),
                0,
                deferredCount,
                0,
                exportAcknowledgements.followUpRequired(),
                exportAcknowledgements.primaryDestination(),
                exportAcknowledgements.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportDecisionsMessage(
                        decisions.size(),
                        deferredCount,
                        exportAcknowledgements.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportResolutions campaignReleaseDeliveryExportResolutions(
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions) {
        List<DiffusionOpdCampaignReleaseDeliveryExportResolution> resolutions = exportDecisions.decisions().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportResolution)
                .toList();
        int deferredCount = resolutions.stream()
                .mapToInt(resolution -> "deferred".equals(resolution.resolution()) ? 1 : 0)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportResolutions(
                exportDecisions,
                resolutions,
                resolutions.size(),
                0,
                deferredCount,
                0,
                exportDecisions.followUpRequired(),
                exportDecisions.primaryDestination(),
                exportDecisions.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportResolutionsMessage(
                        resolutions.size(),
                        deferredCount,
                        exportDecisions.overallDeliveryStatus()));
    }

    static DiffusionOpdCampaignReleaseDeliveryExportClosures campaignReleaseDeliveryExportClosures(
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions) {
        List<DiffusionOpdCampaignReleaseDeliveryExportClosure> closures = exportResolutions.resolutions().stream()
                .map(DiffusionOpdCampaignReleaseDeliveryExportStages::campaignReleaseDeliveryExportClosure)
                .toList();
        int openCount = closures.stream()
                .mapToInt(closure -> closure.closed() ? 0 : 1)
                .sum();
        return new DiffusionOpdCampaignReleaseDeliveryExportClosures(
                exportResolutions,
                closures,
                closures.size(),
                0,
                openCount,
                exportResolutions.followUpRequired(),
                exportResolutions.primaryDestination(),
                exportResolutions.overallDeliveryStatus(),
                DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportClosuresMessage(
                        closures.size(),
                        openCount,
                        exportResolutions.overallDeliveryStatus()));
    }

    static List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> buildCampaignReleaseDeliveryExportHandoffBundles(
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts,
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements,
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions,
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions,
            DiffusionOpdCampaignReleaseDeliveryExportClosures exportClosures) {
        int size = exportReceipts.receipts().size();
        java.util.ArrayList<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> bundles =
                new java.util.ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            DiffusionOpdCampaignReleaseDeliveryExportReceipt receipt = exportReceipts.receipts().get(i);
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgement acknowledgement =
                    exportAcknowledgements.acknowledgements().get(i);
            DiffusionOpdCampaignReleaseDeliveryExportDecision decision = exportDecisions.decisions().get(i);
            DiffusionOpdCampaignReleaseDeliveryExportResolution resolution = exportResolutions.resolutions().get(i);
            DiffusionOpdCampaignReleaseDeliveryExportClosure closure = exportClosures.closures().get(i);
            String handoffTarget = resolution.escalationTarget();
            String exportStatus = closure.finalOutcome();
            boolean terminal = closure.closed();
            bundles.add(new DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle(
                    receipt,
                    acknowledgement,
                    decision,
                    resolution,
                    closure,
                    handoffTarget,
                    exportStatus,
                    terminal,
                    DiffusionOpdReportQueries.buildCampaignReleaseDeliveryExportHandoffBundleMessage(
                            receipt.artifactType(),
                            handoffTarget,
                            exportStatus)));
        }
        return List.copyOf(bundles);
    }

    record CampaignReleaseDeliveryExportPipeline(
            DiffusionOpdCampaignReleaseDeliveryExportManifest exportManifest,
            DiffusionOpdCampaignReleaseDeliveryExportReport exportReport,
            DiffusionOpdCampaignReleaseDeliveryExportLedger exportLedger,
            DiffusionOpdCampaignReleaseDeliveryExportReceipts exportReceipts,
            DiffusionOpdCampaignReleaseDeliveryExportAcknowledgements exportAcknowledgements,
            DiffusionOpdCampaignReleaseDeliveryExportDecisions exportDecisions,
            DiffusionOpdCampaignReleaseDeliveryExportResolutions exportResolutions,
            DiffusionOpdCampaignReleaseDeliveryExportClosures exportClosures,
            List<DiffusionOpdCampaignReleaseDeliveryExportHandoffBundle> handoffBundles) {
    }
}
