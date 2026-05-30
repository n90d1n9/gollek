package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed grouped workflow execution plan derived from a workflow action packet.
 */
public record DiffusionOpdGroupedHistoryWorkflowExecutionPlan(
        DiffusionOpdGroupedHistoryWorkflowActionPacket actionPacket,
        DiffusionOpdWorkflowApprovalGate approvalGate,
        String executionMode,
        Integer cooldownRounds,
        Integer nextCheckWindowSize,
        String escalationTarget,
        List<String> checklist,
        String summaryMessage) {
}
