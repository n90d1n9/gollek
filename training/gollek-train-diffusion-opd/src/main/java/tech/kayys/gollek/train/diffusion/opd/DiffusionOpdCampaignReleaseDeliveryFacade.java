package tech.kayys.gollek.train.diffusion.opd;

import tech.kayys.gollek.train.diffusion.api.DiffusionOpdCampaignReleaseExportManifest;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Small composition entrypoint for release-delivery export and adapter pipelines.
 *
 * <p>This keeps the final release-delivery pipeline wiring out of the public query class while
 * still leaving the public API methods as simple one-line delegators. The facade also keeps the
 * export-manifest root shared so adapter callers do not rebuild the same release export state
 * twice. In practice this file is the boundary between the public query surface and the internal
 * export-plus-adapter helper cluster.
 */
final class DiffusionOpdCampaignReleaseDeliveryFacade {
    private DiffusionOpdCampaignReleaseDeliveryFacade() {
    }

    /**
     * Loads the shared release-export root that both the canonical export lane and compatibility
     * adapter lane build from.
     */
    static DiffusionOpdCampaignReleaseExportManifest buildReleaseExportManifest(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return DiffusionOpdReportQueries.taskCampaignReleaseExportManifest(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit);
    }

    /**
     * Builds the canonical release-delivery export pipeline from a precomputed release-export
     * manifest.
     */
    static DiffusionOpdCampaignReleaseDeliveryExports.CampaignReleaseDeliveryExportPipeline buildExportPipeline(
            DiffusionOpdCampaignReleaseExportManifest exportManifest) {
        return DiffusionOpdCampaignReleaseDeliveryExports.buildCampaignReleaseDeliveryExportPipeline(exportManifest);
    }

    /**
     * Convenience overload for callers that want the canonical export pipeline without explicitly
     * loading the release-export root first.
     */
    static DiffusionOpdCampaignReleaseDeliveryExports.CampaignReleaseDeliveryExportPipeline buildExportPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        return buildExportPipeline(buildReleaseExportManifest(
                baselineReport,
                currentReport,
                shortWindowSize,
                mediumWindowSize,
                longWindowSize,
                limit));
    }

    /**
     * Builds the compatibility adapter pipeline on top of the shared export root and canonical
     * export pipeline so aggregate delivery callers reuse the same underlying release state.
     */
    static DiffusionOpdCampaignReleaseDeliveryAdapters.CampaignReleaseDeliveryAdapterPipeline buildAdapterPipeline(
            DiffusionOpdReport baselineReport,
            DiffusionOpdReport currentReport,
            int shortWindowSize,
            int mediumWindowSize,
            int longWindowSize,
            int limit) {
        DiffusionOpdCampaignReleaseExportManifest exportManifest =
                buildReleaseExportManifest(
                        baselineReport,
                        currentReport,
                        shortWindowSize,
                        mediumWindowSize,
                        longWindowSize,
                        limit);
        DiffusionOpdCampaignReleaseDeliveryExports.CampaignReleaseDeliveryExportPipeline exportPipeline =
                buildExportPipeline(exportManifest);
        return DiffusionOpdCampaignReleaseDeliveryAdapters.buildCampaignReleaseDeliveryAdapterPipeline(
                exportManifest,
                exportPipeline);
    }
}
