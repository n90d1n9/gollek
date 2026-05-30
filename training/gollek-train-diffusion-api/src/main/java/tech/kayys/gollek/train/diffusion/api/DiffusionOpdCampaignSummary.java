package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed cross-portfolio workflow campaign summary for run-level orchestration.
 */
public record DiffusionOpdCampaignSummary(
        DiffusionOpdGroupedHistoryWorkflowBatchManifest groupedBatchManifest,
        DiffusionOpdPairedHistoryWorkflowBatchManifest pairedBatchManifest,
        int batchCount,
        int envelopeCount,
        int targetSliceCount,
        int approvalRequiredCount,
        int manualReviewCount,
        String highestDispatchPriority,
        Integer earliestRecheckAfterRounds,
        Integer earliestDeferredRecheckWindowSize,
        String campaignKey,
        String summaryMessage) {
}
