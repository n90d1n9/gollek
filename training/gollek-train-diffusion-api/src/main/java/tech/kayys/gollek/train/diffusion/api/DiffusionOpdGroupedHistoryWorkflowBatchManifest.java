package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped workflow batch manifest for scheduler-facing execution bundles.
 */
public record DiffusionOpdGroupedHistoryWorkflowBatchManifest(
        List<DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope> envelopes,
        int envelopeCount,
        int targetSliceCount,
        int approvalRequiredCount,
        int manualReviewCount,
        String highestDispatchPriority,
        Integer earliestRecheckAfterRounds,
        Integer earliestDeferredRecheckWindowSize,
        String batchKey,
        String summaryMessage) {
}
