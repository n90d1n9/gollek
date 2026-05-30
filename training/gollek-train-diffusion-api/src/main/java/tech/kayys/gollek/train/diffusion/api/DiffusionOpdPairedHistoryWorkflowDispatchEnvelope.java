package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired workflow dispatch envelope for scheduler-facing execution.
 */
public record DiffusionOpdPairedHistoryWorkflowDispatchEnvelope(
        DiffusionOpdPairedHistoryWorkflowExecutionPlan executionPlan,
        String firstField,
        String secondField,
        List<String> targetPairs,
        String dispatchPriority,
        String retryPolicy,
        Integer recheckAfterRounds,
        Integer deferredRecheckWindowSize,
        String dispatchKey,
        String summaryMessage) {
}
