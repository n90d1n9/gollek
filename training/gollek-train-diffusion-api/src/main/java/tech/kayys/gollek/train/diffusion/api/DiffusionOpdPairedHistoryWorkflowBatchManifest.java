package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired workflow batch manifest for scheduler-facing execution bundles.
 */
public record DiffusionOpdPairedHistoryWorkflowBatchManifest(
        List<DiffusionOpdPairedHistoryWorkflowDispatchEnvelope> envelopes,
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
