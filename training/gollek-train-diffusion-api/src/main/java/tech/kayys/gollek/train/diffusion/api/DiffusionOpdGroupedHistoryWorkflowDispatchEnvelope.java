package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped workflow dispatch envelope for scheduler-facing execution.
 */
public record DiffusionOpdGroupedHistoryWorkflowDispatchEnvelope(
        DiffusionOpdGroupedHistoryWorkflowExecutionPlan executionPlan,
        String targetField,
        List<String> targetValues,
        String dispatchPriority,
        String retryPolicy,
        Integer recheckAfterRounds,
        Integer deferredRecheckWindowSize,
        String dispatchKey,
        String summaryMessage) {
}
