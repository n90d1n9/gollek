package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed paired workflow execution plan derived from a workflow action packet.
 */
public record DiffusionOpdPairedHistoryWorkflowExecutionPlan(
        DiffusionOpdPairedHistoryWorkflowActionPacket actionPacket,
        DiffusionOpdWorkflowApprovalGate approvalGate,
        String executionMode,
        Integer cooldownRounds,
        Integer nextCheckWindowSize,
        String escalationTarget,
        List<String> checklist,
        String summaryMessage) {
}
