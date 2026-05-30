package tech.kayys.gollek.ml.reasoning;

import java.util.Map;
import java.util.Objects;

/**
 * Compact in-memory report for a recursive reasoning demo/session run.
 */
public record RecursiveReasoningReport(
        String reportVersion,
        String familyId,
        String taskId,
        RecursiveReasoningConfig config,
        RecursiveReasoningRolloutSummary summary,
        String selectedStateId,
        int selectedTrajectoryIndex,
        double selectedCumulativeLogProbability,
        Double selectedRewardScore,
        Map<String, Object> metadata) {

    public RecursiveReasoningReport {
        reportVersion = Objects.requireNonNullElse(reportVersion, "1");
        familyId = Objects.requireNonNullElse(familyId, RecursiveReasoningModelFamily.FAMILY_ID);
        taskId = Objects.requireNonNull(taskId, "taskId must not be null");
        config = Objects.requireNonNull(config, "config must not be null");
        summary = Objects.requireNonNull(summary, "summary must not be null");
        selectedStateId = Objects.requireNonNull(selectedStateId, "selectedStateId must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
